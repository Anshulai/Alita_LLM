/*
 * alita_gpu.cu — GPU-accelerated DialoGPT inference in CUDA C
 * Converted from Alita.c (pure C CPU) → CUDA
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 -o alita_gpu alita_gpu.cu -lm
 *   (GTX 1650 = Turing = sm_75)
 * Run:
 *   ./alita_gpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* ── Config ─────────────────────────────────────────────────────────────── */
#define N_LAYERS   24
#define MAX_CTX    128
#define MAX_NEW    80
#define TOP_K      60
#define CSV_DIR    "dialogpt_csv"
#define MAX_LINE   (1024 * 256)

/* ── CUDA error check ────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static inline int cdiv(int a, int b){ return (a+b-1)/b; }

/* ── Structs ─────────────────────────────────────────────────────────────── */
typedef struct {
    float *data;
    int    rows, cols;
} Mat;

typedef struct {
    float *ln1_w, *ln1_b;
    float *ln2_w, *ln2_b;
    Mat    qkv_w;  float *qkv_b;
    Mat    ap_w;   float *ap_b;
    Mat    fc_w;   float *fc_b;
    Mat    pw_w;   float *pw_b;
    /* GPU mirrors */
    float *d_ln1_w, *d_ln1_b;
    float *d_ln2_w, *d_ln2_b;
    float *d_qkv_w, *d_qkv_b;
    float *d_ap_w,  *d_ap_b;
    float *d_fc_w,  *d_fc_b;
    float *d_pw_w,  *d_pw_b;
} Layer;

typedef struct {
    Mat    token_embed;
    Mat    pos_embed;
    float *ln_f_w, *ln_f_b;
    float *d_token_embed, *d_pos_embed;
    float *d_ln_f_w,      *d_ln_f_b;
    Layer  layers[N_LAYERS];
    int    d_model, n_heads, vocab_size;
} Model;

/* ── Memory helpers ──────────────────────────────────────────────────────── */
static float* alloc_vec(int n){
    float *v=(float*)calloc(n,sizeof(float)); if(!v){fprintf(stderr,"OOM\n");exit(1);} return v;
}
static Mat alloc_mat(int r, int c){
    Mat m; m.rows=r; m.cols=c;
    m.data=(float*)calloc(r*c,sizeof(float)); if(!m.data){fprintf(stderr,"OOM\n");exit(1);} return m;
}
#define M(m,i,j) ((m).data[(i)*(m).cols+(j)])

static float* gpu_upload(float *h, int n){
    float *d; CUDA_CHECK(cudaMalloc(&d,n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d,h,n*sizeof(float),cudaMemcpyHostToDevice)); return d;
}
/* gpu_zeros: kept for potential future use */
__attribute__((unused))
static float* gpu_zeros(int n){
    float *d; CUDA_CHECK(cudaMalloc(&d,n*sizeof(float)));
    CUDA_CHECK(cudaMemset(d,0,n*sizeof(float))); return d;
}

/* ── CSV loader ──────────────────────────────────────────────────────────── */
Mat load_csv(const char *filename){
    char path[512]; snprintf(path,sizeof(path),"%s/%s",CSV_DIR,filename);
    FILE *f=fopen(path,"r"); if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    int rows,cols; if(fscanf(f,"%d,%d\n",&rows,&cols)!=2){fprintf(stderr,"Bad CSV header in %s\n",path);exit(1);}
    Mat m=alloc_mat(rows,cols);
    char *line=(char*)malloc(MAX_LINE);
    for(int i=0;i<rows;i++){
        if(!fgets(line,MAX_LINE,f)) break;
        char *tok=strtok(line,",\n");
        for(int j=0;j<cols&&tok;j++){M(m,i,j)=(float)atof(tok);tok=strtok(NULL,",\n");}
    }
    free(line); fclose(f); return m;
}
float* load_vec_csv(const char *filename, int *out_len){
    Mat m=load_csv(filename); *out_len=m.cols; return m.data;
}

/* ── CUDA kernels ────────────────────────────────────────────────────────── */

/* Layer norm: one block per token.
 * Launch with blockDim = min(512, next_pow2(d)), shared = blockDim*sizeof(float).
 * Each thread handles multiple elements when d > blockDim. */
__global__ void kernel_layer_norm(const float *x, const float *w, const float *b,
                                   float *out, int d){
    extern __shared__ float smem[];
    int t   = blockIdx.x;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    const float *xi = x + t*d;
    float *oi = out + t*d;

    /* Sum for mean — each thread covers its stripe */
    float psum = 0.f;
    for(int i = tid; i < d; i += bsz) psum += xi[i];
    smem[tid] = psum; __syncthreads();
    for(int s = bsz/2; s > 0; s >>= 1){
        if(tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    float mean = smem[0] / d; __syncthreads();

    /* Sum for variance */
    float vsum = 0.f;
    for(int i = tid; i < d; i += bsz){ float v = xi[i]-mean; vsum += v*v; }
    smem[tid] = vsum; __syncthreads();
    for(int s = bsz/2; s > 0; s >>= 1){
        if(tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0]/d + 1e-5f);

    /* Write normalized output */
    for(int i = tid; i < d; i += bsz)
        oi[i] = (xi[i]-mean)*inv_std * w[i] + b[i];
}

/* Batch vecmat: Out[t] = V[t] @ M  (T tokens) */
__global__ void kernel_batch_vecmat(const float *V, const float *M, float *Out,
                                     int T, int d_in, int d_out){
    int t=blockIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=T||j>=d_out) return;
    const float *v=V+t*d_in; float s=0.f;
    for(int i=0;i<d_in;i++) s+=v[i]*M[i*d_out+j];
    Out[t*d_out+j]=s;
}

/* Single vecmat: out = v @ M */
__global__ void kernel_vecmat(const float *v, const float *M, float *out,
                               int d_in, int d_out){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=d_out) return;
    float s=0.f;
    for(int i=0;i<d_in;i++) s+=v[i]*M[i*d_out+j];
    out[j]=s;
}

/* Add bias in-place */
__global__ void kernel_add_bias(float *X, const float *b, int T, int d){
    int t=blockIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=T||j>=d) return;
    X[t*d+j]+=b[j];
}

/* Residual add in-place: a += b */
__global__ void kernel_add(float *a, const float *b, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) a[i]+=b[i];
}

/* GELU in-place */
__global__ void kernel_gelu(float *x, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float v=x[i];
    x[i]=0.5f*v*(1.f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));
}

/* Logits = h(1×d) @ embed.T(d×vs) */
__global__ void kernel_logits(const float *h, const float *embed,
                               float *logits, int vs, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=vs) return;
    float s=0.f;
    for(int k=0;k<d;k++) s+=h[k]*embed[j*d+k];
    logits[j]=s;
}

/* Embedding lookup + positional */
__global__ void kernel_embed_one(const float *tok_e, const float *pos_e,
                                  int tok_id, int pos, float *out, int d){
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k<d) out[k]=tok_e[tok_id*d+k]+pos_e[pos*d+k];
}

/* ── Load model ──────────────────────────────────────────────────────────── */
Model* load_model(){
    Model *m=(Model*)calloc(1,sizeof(Model));
    printf("Loading model weights...\n");

    char path[512];
    snprintf(path,sizeof(path),"%s/config.json",CSV_DIR);
    FILE *cfg=fopen(path,"r"); if(!cfg){fprintf(stderr,"config.json not found\n");exit(1);}
    char buf[2048]; memset(buf,0,sizeof(buf));
    size_t bread=fread(buf,1,sizeof(buf)-1,cfg); fclose(cfg);
    (void)bread; /* silence unused-result warning; buf is null-terminated above */

    char *p;
    p=strstr(buf,"\"d_model\":");    m->d_model   =p?atoi(p+10):1024;
    p=strstr(buf,"\"n_heads\":");    m->n_heads   =p?atoi(p+10):16;
    p=strstr(buf,"\"vocab_size\":"); m->vocab_size=p?atoi(p+13):50257;
    printf("  d_model=%d  n_heads=%d  vocab=%d  layers=%d\n",
           m->d_model,m->n_heads,m->vocab_size,N_LAYERS);

    /* Embeddings */
    m->token_embed=load_csv("token_embed.csv");
    m->pos_embed  =load_csv("pos_embed.csv");
    int tmp;
    m->ln_f_w=load_vec_csv("ln_f_weight.csv",&tmp);
    m->ln_f_b=load_vec_csv("ln_f_bias.csv",&tmp);

    /* Upload embeddings to GPU */
    int d=m->d_model;
    m->d_token_embed=gpu_upload(m->token_embed.data, m->vocab_size*d);
    m->d_pos_embed  =gpu_upload(m->pos_embed.data,   m->pos_embed.rows*d);
    m->d_ln_f_w     =gpu_upload(m->ln_f_w, d);
    m->d_ln_f_b     =gpu_upload(m->ln_f_b, d);
    printf("  Embeddings loaded + uploaded to GPU\n");

    /* Layers */
    char fname[128];
    for(int i=0;i<N_LAYERS;i++){
        Layer *l=&m->layers[i];
        #define LV(field,file) \
            snprintf(fname,sizeof(fname),file,i); \
            l->field=load_vec_csv(fname,&tmp); \
            l->d_##field=gpu_upload(l->field,tmp)
        #define LM(field,file) \
            snprintf(fname,sizeof(fname),file,i); \
            l->field=load_csv(fname); \
            l->d_##field=gpu_upload(l->field.data, l->field.rows*l->field.cols)

        LV(ln1_w,"h%d_ln1_w.csv"); LV(ln1_b,"h%d_ln1_b.csv");
        LV(ln2_w,"h%d_ln2_w.csv"); LV(ln2_b,"h%d_ln2_b.csv");
        LM(qkv_w,"h%d_attn_qkv_w.csv");
        LV(qkv_b,"h%d_attn_qkv_b.csv");
        LM(ap_w,"h%d_attn_proj_w.csv");
        LV(ap_b,"h%d_attn_proj_b.csv");
        LM(fc_w,"h%d_mlp_fc_w.csv");
        LV(fc_b,"h%d_mlp_fc_b.csv");
        LM(pw_w,"h%d_mlp_proj_w.csv");
        LV(pw_b,"h%d_mlp_proj_b.csv");
        #undef LV
        #undef LM
        printf("  Layer %d/%d loaded\r",i+1,N_LAYERS); fflush(stdout);
    }
    printf("  All %d layers loaded + uploaded to GPU!          \n",N_LAYERS);
    return m;
}

/* ── GPU Forward pass ────────────────────────────────────────────────────── */
/*
 * Architecture decision (same as train_gpu.cu):
 *   - Embedding, layer norm, QKV/MLP projections → GPU kernels
 *   - Causal multi-head attention (small ctx) → CPU (avoid complex GPU scatter)
 *   - Logits computation → GPU kernel (vocab * d_model dot products)
 *
 * All intermediate states live in pinned/device memory; attention buffers
 * are small (<128 tokens × 1024) and fast to transfer.
 */
void forward_gpu(Model *m, int *tokens, int n_tokens, float *logits_out){
    int d       = m->d_model;
    int n_heads = m->n_heads;
    int hd      = d / n_heads;
    int vs      = m->vocab_size;

    /* --- Build input embeddings on GPU --- */
    float *d_x; CUDA_CHECK(cudaMalloc(&d_x, n_tokens*d*sizeof(float)));
    for(int t=0;t<n_tokens;t++){
        int tid=tokens[t]; if(tid<0)tid=0; if(tid>=vs)tid=vs-1;
        int pos=(t<m->pos_embed.rows)?t:m->pos_embed.rows-1;
        kernel_embed_one<<<cdiv(d,128),128>>>(
            m->d_token_embed, m->d_pos_embed, tid, pos, d_x+t*d, d);
    }

    /* Scratch buffers — sized for ALL tokens, not just one */
    float *d_xn, *d_qkv, *d_proj, *d_ff, *d_attn_gpu;
    CUDA_CHECK(cudaMalloc(&d_xn,      n_tokens*d*sizeof(float)));   /* was: d only — BUG FIXED */
    CUDA_CHECK(cudaMalloc(&d_qkv,     n_tokens*3*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj,    n_tokens*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff,      n_tokens*d*4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_gpu,n_tokens*d*sizeof(float)));   /* was: d only — BUG FIXED */

    /* CPU-side QKV buffer for causal attention */
    float *h_qkv=(float*)malloc(n_tokens*3*d*sizeof(float));
    float *h_x  =(float*)malloc(n_tokens*d*sizeof(float));
    float *h_attn=(float*)malloc(n_tokens*d*sizeof(float));

    /* blk_ln: power-of-2, capped at 512 (GPU max threads/block is 1024,
     * but 512 is safe and leaves headroom; kernel uses strided loop for d>512) */
    int blk_ln = 1;
    while(blk_ln < d && blk_ln < 512) blk_ln <<= 1;

    for(int li=0;li<N_LAYERS;li++){
        Layer *l=&m->layers[li];

        /* LN1 + QKV for all positions */
        kernel_layer_norm<<<n_tokens, blk_ln, blk_ln*sizeof(float)>>>(
            d_x, l->d_ln1_w, l->d_ln1_b, d_xn, d);
        /* We run QKV per-token and store */
        dim3 grd_q(cdiv(3*d,128),n_tokens);
        kernel_batch_vecmat<<<grd_q,128>>>(d_xn, l->d_qkv_w, d_qkv, n_tokens, d, 3*d);
        kernel_add_bias<<<dim3(cdiv(3*d,128),n_tokens),128>>>(d_qkv, l->d_qkv_b, n_tokens, 3*d);

        /* Download QKV + current X for attention */
        CUDA_CHECK(cudaMemcpy(h_qkv, d_qkv, n_tokens*3*d*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_x,   d_x,   n_tokens*d*sizeof(float),   cudaMemcpyDeviceToHost));

        /* --- Causal multi-head attention (CPU) --- */
        float *scores=(float*)malloc(n_tokens*sizeof(float));
        for(int pos=0;pos<n_tokens;pos++){
            memset(h_attn+pos*d, 0, d*sizeof(float));
            for(int h=0;h<n_heads;h++){
                int hs=h*hd;
                float sc=sqrtf((float)hd);
                float *Q=h_qkv+pos*3*d+hs;
                for(int j=0;j<=pos;j++){
                    float *K=h_qkv+j*3*d+d+hs;
                    float s=0; for(int k=0;k<hd;k++) s+=Q[k]*K[k];
                    scores[j]=s/sc;
                }
                float mx=scores[0]; for(int j=1;j<=pos;j++) if(scores[j]>mx) mx=scores[j];
                float sm=0; for(int j=0;j<=pos;j++){scores[j]=expf(scores[j]-mx);sm+=scores[j];}
                for(int j=0;j<=pos;j++) scores[j]/=sm;
                for(int j=0;j<=pos;j++){
                    float w=scores[j];
                    float *V=h_qkv+j*3*d+2*d+hs;
                    for(int k=0;k<hd;k++) h_attn[pos*d+hs+k]+=w*V[k];
                }
            }
        }
        free(scores);

        /* Upload attention output, run projection + residual on GPU */
        CUDA_CHECK(cudaMemcpy(d_attn_gpu, h_attn, n_tokens*d*sizeof(float), cudaMemcpyHostToDevice));

        /* Re-use d_xn as output of projection for each token */
        /* For n_tokens, do batch ap projection */
        float *d_ap_out; CUDA_CHECK(cudaMalloc(&d_ap_out, n_tokens*d*sizeof(float)));
        dim3 grd_ap(cdiv(d,128),n_tokens);
        kernel_batch_vecmat<<<grd_ap,128>>>(d_attn_gpu, l->d_ap_w, d_ap_out, n_tokens, d, d);
        kernel_add_bias<<<dim3(cdiv(d,128),n_tokens),128>>>(d_ap_out, l->d_ap_b, n_tokens, d);
        kernel_add<<<cdiv(n_tokens*d,256),256>>>(d_x, d_ap_out, n_tokens*d);
        CUDA_CHECK(cudaFree(d_ap_out));

        /* LN2 */
        kernel_layer_norm<<<n_tokens, blk_ln, blk_ln*sizeof(float)>>>(
            d_x, l->d_ln2_w, l->d_ln2_b, d_xn, d);

        /* MLP fc */
        int d_ff_sz=l->fc_w.cols;
        float *d_hff; CUDA_CHECK(cudaMalloc(&d_hff, n_tokens*d_ff_sz*sizeof(float)));
        dim3 grd_fc(cdiv(d_ff_sz,128),n_tokens);
        kernel_batch_vecmat<<<grd_fc,128>>>(d_xn, l->d_fc_w, d_hff, n_tokens, d, d_ff_sz);
        kernel_add_bias<<<dim3(cdiv(d_ff_sz,128),n_tokens),128>>>(d_hff, l->d_fc_b, n_tokens, d_ff_sz);
        kernel_gelu<<<cdiv(n_tokens*d_ff_sz,256),256>>>(d_hff, n_tokens*d_ff_sz);

        /* MLP pw */
        float *d_ffout; CUDA_CHECK(cudaMalloc(&d_ffout, n_tokens*d*sizeof(float)));
        dim3 grd_pw(cdiv(d,128),n_tokens);
        kernel_batch_vecmat<<<grd_pw,128>>>(d_hff, l->d_pw_w, d_ffout, n_tokens, d_ff_sz, d);
        kernel_add_bias<<<dim3(cdiv(d,128),n_tokens),128>>>(d_ffout, l->d_pw_b, n_tokens, d);
        kernel_add<<<cdiv(n_tokens*d,256),256>>>(d_x, d_ffout, n_tokens*d);
        CUDA_CHECK(cudaFree(d_hff)); CUDA_CHECK(cudaFree(d_ffout));
    }

    /* Final LN on last token only — point into d_x at the last token row */
    float *d_last = d_x + (n_tokens-1)*d;
    float *d_lnf_out; CUDA_CHECK(cudaMalloc(&d_lnf_out, d*sizeof(float)));
    kernel_layer_norm<<<1, blk_ln, blk_ln*sizeof(float)>>>(
        d_last, m->d_ln_f_w, m->d_ln_f_b, d_lnf_out, d);

    /* Logits on GPU */
    float *d_logits; CUDA_CHECK(cudaMalloc(&d_logits, vs*sizeof(float)));
    kernel_logits<<<cdiv(vs,256),256>>>(d_lnf_out, m->d_token_embed, d_logits, vs, d);
    CUDA_CHECK(cudaMemcpy(logits_out, d_logits, vs*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_lnf_out)); CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_xn));
    CUDA_CHECK(cudaFree(d_qkv)); CUDA_CHECK(cudaFree(d_proj));
    CUDA_CHECK(cudaFree(d_ff)); CUDA_CHECK(cudaFree(d_attn_gpu));
    free(h_qkv); free(h_x); free(h_attn);
}

/* ── Top-k sampling (CPU) ────────────────────────────────────────────────── */
int top_k_sample(float *logits, int vocab, float temp, int k){
    for(int i=0;i<vocab;i++) logits[i]/=temp;
    int *idx=(int*)malloc(vocab*sizeof(int));
    for(int i=0;i<vocab;i++) idx[i]=i;
    for(int i=0;i<k;i++){
        int best=i;
        for(int j=i+1;j<vocab;j++) if(logits[idx[j]]>logits[idx[best]]) best=j;
        int tmp=idx[i];idx[i]=idx[best];idx[best]=tmp;
    }
    float mx=logits[idx[0]];
    float *probs=(float*)malloc(k*sizeof(float)); float sum=0;
    for(int i=0;i<k;i++){probs[i]=expf(logits[idx[i]]-mx);sum+=probs[i];}
    for(int i=0;i<k;i++) probs[i]/=sum;
    float r=(float)rand()/RAND_MAX, cum=0;
    int chosen=idx[k-1];
    for(int i=0;i<k;i++){cum+=probs[i];if(r<=cum){chosen=idx[i];break;}}
    free(idx); free(probs); return chosen;
}

/* ── Tokenizer (CPU) ─────────────────────────────────────────────────────── */
#define MAX_VOCAB_T 51000
#define MAX_WORD    64

char  vocab_words[MAX_VOCAB_T][MAX_WORD];
int   vocab_ids  [MAX_VOCAB_T];
char  id_to_word [MAX_VOCAB_T][MAX_WORD];
int   vocab_size_loaded=0;
int   eos_id=50256;

void load_vocab(){
    char path[256]; snprintf(path,sizeof(path),"%s/vocab.json",CSV_DIR);
    FILE *f=fopen(path,"r"); if(!f){fprintf(stderr,"vocab.json not found\n");exit(1);}
    int c,state=0,vi=0; char word[MAX_WORD];int wi=0; char numstr[16];int ni=0;
    while((c=fgetc(f))!=EOF&&vi<MAX_VOCAB_T){
        if(state==0){if(c=='"'){wi=0;state=1;}}
        else if(state==1){if(c=='"'){word[wi]=0;state=2;}else if(wi<MAX_WORD-1)word[wi++]=c;}
        else if(state==2){if(c==':'){ni=0;state=3;}}
        else if(state==3){
            if(c>='0'&&c<='9'){if(ni<15)numstr[ni++]=c;}
            else if(ni>0){
                numstr[ni]=0;int id=atoi(numstr);
                strncpy(vocab_words[vi],word,MAX_WORD-1);vocab_ids[vi]=id;
                if(id<MAX_VOCAB_T)strncpy(id_to_word[id],word,MAX_WORD-1);
                if(strcmp(word,"<|endoftext|>")==0)eos_id=id;
                vi++;ni=0;state=0;
            }
        }
    }
    vocab_size_loaded=vi; fclose(f);
    printf("  Vocab loaded: %d tokens\n",vocab_size_loaded);
}

int encode_word(const char *word){
    char escaped[MAX_WORD+8]; snprintf(escaped,sizeof(escaped),"\\u0120%s",word);
    for(int i=0;i<vocab_size_loaded;i++){
        if(strcmp(vocab_words[i],escaped)==0) return vocab_ids[i];
        if(strcmp(vocab_words[i],word)==0)    return vocab_ids[i];
    }
    char lower[MAX_WORD];int li=0;
    while(word[li]){lower[li]=(word[li]>='A'&&word[li]<='Z')?word[li]+32:word[li];li++;}
    lower[li]=0;
    snprintf(escaped,sizeof(escaped),"\\u0120%s",lower);
    for(int i=0;i<vocab_size_loaded;i++){
        if(strcmp(vocab_words[i],escaped)==0) return vocab_ids[i];
        if(strcmp(vocab_words[i],lower)==0)   return vocab_ids[i];
    }
    return 1;
}
int encode(const char *text, int *out, int max_tokens){
    char buf[1024]; strncpy(buf,text,1023); int n=0;
    char *tok=strtok(buf," \t\n");
    while(tok&&n<max_tokens){out[n++]=encode_word(tok);tok=strtok(NULL," \t\n");}
    return n;
}
void decode_token(int id, char *out){
    if(id<0||id>=MAX_VOCAB_T){out[0]=0;return;}
    const char *w=id_to_word[id];
    if(strlen(w)>=6&&w[0]=='\\'&&w[1]=='u'&&w[2]=='0'&&w[3]=='1'&&w[4]=='2'&&w[5]=='0'){
        out[0]=' '; strcpy(out+1,w+6); return;
    }
    if((unsigned char)w[0]==0xC4&&(unsigned char)w[1]==0xA0){out[0]=' ';strcpy(out+1,w+2);return;}
    if(w[0]==' ') strcpy(out,w+1); else strcpy(out,w);
}

/* ── Main chat loop ──────────────────────────────────────────────────────── */
int main(){
    srand(time(NULL));

    /* GPU info */
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
    printf("GPU: %s  (%.0f MB, sm_%d%d)\n\n",
           prop.name, prop.totalGlobalMem/1e6f, prop.major, prop.minor);

    char check[256]; snprintf(check,sizeof(check),"%s/config.json",CSV_DIR);
    FILE *fc=fopen(check,"r");
    if(!fc){printf("ERROR: %s/ not found.\nRun: python convert_dialogpt.py\n",CSV_DIR);return 1;}
    fclose(fc);

    printf("\nLoading DialoGPT weights from CSV → GPU...\n");
    load_vocab();
    Model *model=load_model();

    float *logits=alloc_vec(model->vocab_size);
    int history[512]; int hist_len=0;

    printf("\n");
    printf("=======================================================\n");
    printf("  Alita  —  DialoGPT in CUDA C (GPU-accelerated!)\n");
    printf("  %d layers | context=%d | GPU: %s\n",N_LAYERS,MAX_CTX,prop.name);
    printf("  type 'exit' to quit\n");
    printf("=======================================================\n\n");

    char input[512];
    while(1){
        printf("You: "); fflush(stdout);
        if(!fgets(input,sizeof(input),stdin)) break;
        int len=strlen(input); if(len>0&&input[len-1]=='\n') input[--len]=0;
        if(strcmp(input,"exit")==0||strcmp(input,"quit")==0){printf("Alita: Goodbye!\n");break;}
        if(len==0) continue;

        int user_ids[64]; int user_len=encode(input,user_ids,60);
        user_ids[user_len++]=eos_id;
        for(int i=0;i<user_len&&hist_len<MAX_CTX;i++) history[hist_len++]=user_ids[i];
        if(hist_len>MAX_CTX){
            int overflow=hist_len-MAX_CTX;
            memmove(history,history+overflow,(hist_len-overflow)*sizeof(int));
            hist_len-=overflow;
        }

        printf("Alita: "); fflush(stdout);
        for(int step=0;step<MAX_NEW;step++){
            forward_gpu(model,history,hist_len,logits);

            for(int i=hist_len-20;i>=0&&i<hist_len;i++){
                int tok=history[i];
                if(tok>=0&&tok<model->vocab_size) logits[tok]-=1.35f;
            }
            int next=top_k_sample(logits,model->vocab_size,1.15f,TOP_K);
            if(next==eos_id) break;

            if(hist_len==MAX_CTX){
                memmove(history,history+1,(MAX_CTX-1)*sizeof(int)); hist_len--;
            }
            history[hist_len++]=next;
            char word[MAX_WORD+2]; decode_token(next,word);
            printf("%s",word); fflush(stdout);
        }
        printf("\n\n");
        if(hist_len<MAX_CTX) history[hist_len++]=eos_id;
    }

    free(logits); return 0;
}
