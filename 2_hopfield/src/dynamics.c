/*******************************************************************************
 *
 *  FILE:         dynamics.c
 *
 *  DATE:         24 February 2023
 *
 *  AUTHORS:      Louis Kang, University of Pennsylvania
 *
 *  LICENSING:  
 *
 *  REFERENCE:  
 *
 *  PURPOSE:      Simulating a Hopfield model with dual encodings. The
 *                inhibition level can vary over time.
 *
 *  DEPENDENCIES: Intel oneAPI MKL
 *
 ******************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

#include "mkl.h"

//==============================================================================
// Parameters and global variables
//==============================================================================


// Random ----------------------------------------------------------------------
uint32_t r4seed = 0;
#define RUNI r4_uni(&r4seed)

// Iterations and time ---------------------------------------------------------
int T_sim = 100;
int T_rec = 10;
int T_subcycle = 0;
enum sequence_type {sequential, block_random, full_random};
enum sequence_type update_type = block_random;

// Network ---------------------------------------------------------------------
int N;
int p = 1;        
int s = 10;      
int N_bin;
int p_max;
int s_max;
float g = 0.1;
float beta = 0.;
float a_sparse;
float a_dense;
float a_fac;

float *w;
float f = 0.;     // strength of cue as input during retrieval

// Theta -----------------------------------------------------------------------
float *theta;
enum function_type {ramp, sine, square, file};
enum function_type theta_type = ramp;
float theta_start = 0.6;
float theta_end = 0.6;
int T_theta = 6;
char theta_name[256];
int cycle_theta = 0;

// Pattern ---------------------------------------------------------------------
char X_dir[256] = "";
int n_cue = 100;        
enum pattern_type {sparse, dense, cat};
float incomp = 0.;
float inacc = 0.;

int sparse_file, dense_file, cat_file;
int *q_mu, *q_nu, *q_cue;
int mu_cue, nu_cue;
unsigned char ***X_sparse, **X_dense, **X_cat;
unsigned char *X_cue, *X_bin;

// Recording -------------------------------------------------------------------
FILE *S_file, *a_file;
FILE *overlap_sparse_file, *overlap_cat_file, *overlap_other_file;
unsigned char *S_bin;
double *overlap_smean, *overlap_cmean;

enum overlap_type {classic, custom};
enum overlap_type record_type = classic;

// I/O -------------------------------------------------------------------------
char fileroot[256] = "";
int save_activity = 0;
int stat_screen = 1;
int stat_log = 1;
FILE *stat_file;

//==============================================================================
// END Parameters and global variables
//==============================================================================



//==============================================================================
// I/O utility functions
//==============================================================================

void print_screen (char *format, ...) {
  
  va_list args1;

  va_start(args1, format);

  if (stat_screen) {
    vprintf(format, args1);
    fflush(stdout);
  }

  va_end(args1);

}


void clear_screen() {

  print_screen("                                                  \r");

}


void print_stat (char *format, ...) {
  
  va_list args1, args2;

  va_start(args1, format);
  va_copy(args2, args1);

  if (stat_screen) {
    vprintf(format, args1);
    fflush(stdout);
  }

  if (stat_log) {
    vfprintf(stat_file, format, args2);
    fflush(stat_file);
  }

  va_end(args1);
  va_end(args2);

}


void print_err (char *format, ...) {
  
  va_list args1, args2;

  va_start(args1, format);
  va_copy(args2, args1);

  vfprintf(stderr, format, args1);
  fflush(stderr);

  if (stat_log) {
    vfprintf(stat_file, format, args2);
    fflush(stat_file);
  }

  va_end(args1);
  va_end(args2);

}


static inline unsigned char rotl (unsigned char c, unsigned int n) {

  n &= 7;
  return (c << n) | (c >> ((-n) & 7));

}

static inline unsigned char rotr (unsigned char c, unsigned int n) {

  n &= 7;
  return (c >> n) | (c << ((-n) & 7));

}

void compress_binary_char (
    unsigned char *raw, unsigned char *bin, int n_raw
) {

  int i;

  for (i = 0; i < n_raw; i += 8)
    bin[i/8] = 0;

  for (i = 0; i < n_raw; i++)
    bin[i/8] |= raw[i] << (7 - (i % 8));

}

void uncompress_binary_char (
    unsigned char *bin, unsigned char *raw, int n_raw
) {

  int i;

  for (i = 0; i < n_raw; i++) {
    bin[i/8] = rotl(bin[i/8], 1);  
    raw[i] = bin[i/8] & 1;
  }

}

void output_int_list(int *list, int n, char *extname) {

  char name[256];
  FILE *file;
  int i;

  sprintf(name, "%s_%s.txt", fileroot, extname);
  file = fopen(name, "w");

  for (i = 0; i < n; i++)
    fprintf(file, "%d ", list[i]); 

  fclose(file);

}

void output_float_list(float *list, int n, char *extname) {

  char name[256];
  FILE *file;
  int i;

  sprintf(name, "%s_%s.txt", fileroot, extname);
  file = fopen(name, "w");

  for (i = 0; i < n; i++)
    fprintf(file, "%f ", list[i]); 

  fclose(file);

}

void output_float_array(float **arr, int ni, int nj, char *extname) {

  char name[256];
  FILE *file;
  int i, j;

  sprintf(name, "%s_%s.txt", fileroot, extname);
  file = fopen(name, "w");

  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) 
      fprintf(file, "%e ", arr[i][j]); 
    fprintf(file, "\n");
  }
  fclose(file);

}

void output_float_1darray(float *arr, int ni, int nj, int incj, char *extname) {

  char name[256];
  FILE *file;
  int i, j;

  sprintf(name, "%s_%s.txt", fileroot, extname);
  file = fopen(name, "w");

  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) 
      fprintf(file, "%e ", arr[i*incj+j]); 
    fprintf(file, "\n");
  }
  fclose(file);

}


//==============================================================================
// END I/O utility functions
//==============================================================================



//==============================================================================
// Math utility functions
//==============================================================================

static inline int fmini (int a, int b) {

  return (a < b) ? a : b;

}

static inline int fmaxi (int a, int b) {

  return (a > b) ? a : b;
}

static inline unsigned char step (float a) { 
  
  return a < 0. ? 0 : 1;

} 

static inline void swap (int *a, int *b) { 

    int temp;

    temp = *a; 
    *a = *b; 
    *b = temp; 

} 

void permutation (int n, int *arr) {
  
  int i, j;

  for (i = 0; i < n; i++)
    arr[i] = i;

  for (i = n-1; i > 0; i--) {
    j = (int) ((double)RUNI * (i+1));
    swap(&arr[i], &arr[j]);
  }

}

void sample (int n_pop, int n_samp, int *arr) {

  int i_pop = 0;
  int i_samp = 0;

  if (n_samp > n_pop) {

    print_err("Error: n_samp cannot be greater than n_pop ");
    print_err("in sample function\n");
    exit(1);

  } else if (n_samp == n_pop) {

     for (i_samp = 0; i_samp < n_samp; i_samp++)
       arr[i_samp] = i_samp;

  } else {

    while (i_samp < n_samp) {
      if ( (n_pop-i_pop) * RUNI >= n_samp-i_samp )
        i_pop++;
      else {
        arr[i_samp] = i_pop;
        i_pop++; i_samp++;
      }
    }

  }

}

float r4_uni (uint32_t *jsr) {

  uint32_t jsr_input;
  float value;

  jsr_input = *jsr;

  *jsr = ( *jsr ^ ( *jsr <<   13 ) );
  *jsr = ( *jsr ^ ( *jsr >>   17 ) );
  *jsr = ( *jsr ^ ( *jsr <<    5 ) );

  value = fmod ( 0.5 
    + ( float ) ( jsr_input + *jsr ) / 65536.0 / 65536.0, 1.0 );

  return value;
}

//==============================================================================
// END Math utility functions
//==============================================================================



//==============================================================================
// Parameters
//==============================================================================

void read_parameters (int argc, char *argv[]) {

  int narg = 1;

  while (narg < argc) {
    if (!strcmp(argv[narg],"-fileroot")) {
      sscanf(argv[narg+1],"%s",fileroot);
      narg += 2;
    } else if (!strcmp(argv[narg],"-seed")) {
      sscanf(argv[narg+1],"%d",&r4seed);
      narg += 2;
    } 

    else if (!strcmp(argv[narg],"-T_sim")) {
      sscanf(argv[narg+1],"%d",&T_sim);
      narg += 2;
    } else if (!strcmp(argv[narg],"-T_rec")) {
      sscanf(argv[narg+1],"%d",&T_rec);
      narg += 2;
    } else if (!strcmp(argv[narg],"-sequential")) {
      update_type = sequential;
      narg++;
    } else if (!strcmp(argv[narg],"-block_random")) {
      update_type = block_random;
      narg++;
    } else if (!strcmp(argv[narg],"-full_random")) {
      update_type = full_random;
      narg++;
    }

    else if (!strcmp(argv[narg],"-N")) {
      sscanf(argv[narg+1],"%d",&N);
      narg += 2;
    } else if (!strcmp(argv[narg],"-p")) {
      sscanf(argv[narg+1],"%d",&p);
      narg += 2;
    } else if (!strcmp(argv[narg],"-s")) {
      sscanf(argv[narg+1],"%d",&s);
      narg += 2;
    } else if (!strcmp(argv[narg],"-gamma")) {
      sscanf(argv[narg+1],"%f",&g);
      narg += 2;
    } else if (!strcmp(argv[narg],"-beta")) {
      sscanf(argv[narg+1],"%f",&beta);
      narg += 2;
    } else if (!strcmp(argv[narg],"-field")) {
      sscanf(argv[narg+1],"%f",&f);
      narg += 2;
    } else if (!strcmp(argv[narg],"-ramp")) {
      theta_type = ramp;
      narg++;
    } else if (!strcmp(argv[narg],"-sine")) {
      theta_type = sine;
      narg++;
    } else if (!strcmp(argv[narg],"-square")) {
      theta_type = square;
      narg++;
    } else if (!strcmp(argv[narg],"-theta_range")) {
      sscanf(argv[narg+1],"%f",&theta_start);
      sscanf(argv[narg+2],"%f",&theta_end);
      sscanf(argv[narg+3],"%d",&T_theta);
      narg += 4;
    } else if (!strcmp(argv[narg],"-theta_file")) {
      sscanf(argv[narg+1],"%s",theta_name);
      theta_type = file;
      narg += 2;
    } else if (!strcmp(argv[narg],"-cycle_theta")) {
      cycle_theta = 1;
      narg++;
    }

    else if (!strcmp(argv[narg],"-X_dir")) {
      sscanf(argv[narg+1],"%s",X_dir);
      narg += 2;
    } else if (!strcmp(argv[narg],"-n_cue")) {
      sscanf(argv[narg+1],"%d",&n_cue);
      narg += 2;
    } else if (!strcmp(argv[narg],"-incomp")) {
      sscanf(argv[narg+1],"%f",&incomp);
      narg += 2;
    } else if (!strcmp(argv[narg],"-inacc")) {
      sscanf(argv[narg+1],"%f",&inacc);
      narg += 2;
    } else if (!strcmp(argv[narg],"-single_flip")) {
      inacc = -1.;
      narg++;
    } else if (!strcmp(argv[narg],"-noiseless")) {
      incomp = 0.;
      inacc = 0.;
      narg++;
    }

    else if (!strcmp(argv[narg],"-verbose")) {
      stat_screen = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-quiet")) {
      stat_screen = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-no_log")) {
      stat_log = 0;
      narg++;
    }

    else if (!strcmp(argv[narg],"-save_activity")) {
      save_activity = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-no_activity")) {
      save_activity = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-custom_overlap")) {
      record_type = custom;
      narg++;
    } else if (!strcmp(argv[narg],"-classic_overlap")) {
      record_type = classic;
      narg++;
    }


    else {
      fprintf(stderr, "unknown option: %s\n", argv[narg]);
      fflush(stderr);
      exit(1);
    }
  }  

}


void print_parameters () {

  FILE *file;
  char name[256];

  sprintf(name, "%s_params.txt", fileroot);
  file = fopen(name, "w");

  fprintf(file, "seed            = %d\n", r4seed);

  fprintf(file, "\n");
  fprintf(file, "T_sim           = %d\n", T_sim);
  fprintf(file, "T_rec           = %d\n", T_rec);
  fprintf(file, "update_type     = %d\n", update_type);

  fprintf(file, "\n");
  fprintf(file, "N               = %d\n", N);
  fprintf(file, "p               = %d\n", p);
  fprintf(file, "s               = %d\n", s);
  fprintf(file, "gamma           = %f\n", g);
  fprintf(file, "beta            = %f\n", beta);
  fprintf(file, "field           = %f\n", f);

  fprintf(file, "\n");
  fprintf(file, "theta_type      = %d\n", theta_type);
  fprintf(file, "theta_start     = %f\n", theta_start);
  fprintf(file, "theta_end       = %f\n", theta_end);
  fprintf(file, "T_theta         = %d\n", T_theta);
  fprintf(file, "theta_file      = %s\n", theta_name);
  fprintf(file, "cycle_theta     = %d\n", cycle_theta);

  fprintf(file, "\n");
  fprintf(file, "X_dir           = %s\n", X_dir);
  fprintf(file, "n_cue           = %d\n", n_cue);
  fprintf(file, "incomp          = %f\n", incomp);
  fprintf(file, "inacc           = %f\n", inacc);
  fprintf(file, "record_type     = %d\n", record_type);

  fclose(file);

}


void setup_parameters (int argc, char *argv[]) {
  
  char name[256];
  char *token;
  char a_buf[16];
  int ch, place;


  read_parameters(argc, argv);


  if (strcmp(fileroot, "") == 0 ||
      fileroot[0] == '-' ||
      fileroot[strlen(fileroot)-1] == '/') {
    fprintf(stderr, "Must set -fileroot correctly\n");
    fflush(stderr);
    exit(1);
  }
  if (stat_log) {
    sprintf(name, "%s_log.txt", fileroot);
    stat_file = fopen(name, "w");
  }


  print_stat("\nSTARTING SIMULATION FOR %s\n", fileroot);


  if (X_dir[strlen(X_dir)-1] == '/')
    X_dir[strlen(X_dir)-1] = '\0';
  
  strcpy(name, X_dir);
  token = strtok(name, "/-_.");
  while (token != NULL) {
    if (strlen(token) > 1 && token[1] >= '0' && token[1] <= '9') {
      if (token[0] == 'N')
        sscanf(token, "N%d", &N);
      else if (token[0] == 'p')
        sscanf(token, "p%d", &p_max);
      else if (token[0] == 's')
        sscanf(token, "s%d", &s_max);
      else if (token[0] == 'a') {
        sprintf(a_buf, "0.%s", token+1);
        a_sparse = strtof(a_buf, NULL);
        token = strtok(NULL, "/-.");
        sprintf(a_buf, "0.%s", token);
        a_dense = strtof(a_buf, NULL);
      }
    }
    token = strtok(NULL, "/-_.");
  }
  if (N == 0 || p_max == 0 || s_max == 0) {
    print_err("Error: pattern directory name %s must contain numbers ", X_dir);
    print_err("of neurons, categories, and examples\n");
    exit(1);
  } else if (p > p_max) {
    print_err("Error: requested categories %d exceeds maximum %d\n", p, p_max);
    exit(1);
  } else if (s > s_max) {
    print_err("Error: requested examples %d exceeds maximum %d\n", s, s_max);
    exit(1);
  } else {
    print_stat("Pattern directory contains %d neurons, ", N);
    print_stat("%d categories, and %d examples\n", p_max, s_max);
  }
  if (a_sparse == 0. || a_dense == 0.) {
    print_err("Error: pattern directory name %s must contain ", X_dir);
    print_err("sparsenesses\n");
    exit(1);
  } else {
    print_stat("Sparsenseses detected: %4.3f and %4.3f\n", a_sparse, a_dense);
  }


  if (r4seed == 0) {
    r4seed = (uint32_t)(time(NULL) % 1048576);
    place = 1;
    for (ch = 0; ch < strlen(fileroot); ch++) {
      r4seed += fileroot[ch] * place;
      place = (place * 128) % 100000;
    }
  }

  N_bin = (N-1)/8 + 1;


  print_parameters();

}


//==============================================================================
// END Parameters
//==============================================================================



//==============================================================================
// Setup
//==============================================================================

void read_pattern (unsigned char *X, int file, int mu, int nu) {

  int q;

  if (nu >= 0)
    q = q_mu[mu]*s_max + q_nu[nu];
  else
    q = q_mu[mu];

  if (pread(file, X_bin, N_bin, (long int) q*N_bin) != N_bin) {
    print_err("Error: pattern file too short\n");
    exit(1);
  };
  uncompress_binary_char(X_bin, X, N);

}


void setup_patterns () {

  float *X_combo;
  char sname[256], dname[256], cname[256];

  int i, j, mu, nu;


  print_stat("Storing %d categories with %d examples\n", p, s);

  sprintf(sname, "%s/x_sparse.dat", X_dir);
  sprintf(dname, "%s/x_dense.dat" , X_dir);
  sprintf(cname, "%s/x_cat.dat"   , X_dir);
  if ((sparse_file = open(sname, O_RDONLY)) < 0 ||
      (dense_file  = open(dname, O_RDONLY)) < 0 ||
      (cat_file    = open(cname, O_RDONLY)) < 0   ) {
    print_err("Error opening pattern files in directory %s\n", X_dir);
    exit(1);
  }


  q_mu = malloc(p * sizeof(int));
  sample(p_max, p, q_mu);
  output_int_list(q_mu, p, "qmu");

  q_nu = malloc(s * sizeof(int));
  sample(s_max, s, q_nu);
  output_int_list(q_nu, s, "qnu");


  X_bin = malloc(N_bin * sizeof(char));
  X_cue = malloc(N * sizeof(char));

  X_sparse = malloc(p * sizeof(char **));
  for (mu = 0; mu < p; mu++) {
    X_sparse[mu] = malloc(s * sizeof(char *));
    for (nu = 0; nu < s; nu++)
      X_sparse[mu][nu] = malloc(N * sizeof(char));
  }
  X_dense = malloc(s * sizeof(char *));
  for (nu = 0; nu < s; nu++)
    X_dense[nu] = malloc(N * sizeof(char));
  X_cat = malloc(p * sizeof(char *));
  for (mu = 0; mu < p; mu++)
    X_cat[mu] = malloc(N * sizeof(char));

  X_combo = mkl_malloc(s*N * sizeof(float), 64);
  w = mkl_calloc(N*N, sizeof(float *), 64);


  for (mu = 0; mu < p; mu++) {

    print_screen("%3d/%3d categories stored...\r", mu, p);

    read_pattern(X_cat[mu], cat_file, mu, -1);

    for (nu = 0; nu < s; nu++) {

      read_pattern(X_sparse[mu][nu], sparse_file, mu, nu);
      read_pattern(X_dense     [nu], dense_file , mu, nu);

    }

    for (nu = 0; nu < s; nu++)
      for (i = 0; i < N; i++)
        X_combo[nu*N+i] =   (1.-2.*g) * (X_sparse[mu][nu][i] - a_sparse)
                          + (   2.*g) * (X_dense     [nu][i] - a_dense );

    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans,
                N, s, 1./N, X_combo, N, 1., w, N     );

  } 

  clear_screen();

  for (i = 0; i < N; i++) {
    w[i*N+i] = 0.;
    for (j = i; j < N; j++)
      w[j*N+i] = w[i*N+j];
  }
 
  output_float_1darray(w, fmini(N,100), fmini(N,100), N, "wsample");

  free(X_bin);
  for (nu = 0; nu < s; nu++)
    free(X_dense[nu]);
  free(X_dense);
  mkl_free(X_combo);

  close(sparse_file);
  close(dense_file);
  close(cat_file);
  

  a_fac = (1.-2.*g)*(1.-2.*g) * a_sparse;
  beta /= a_fac;

}




void setup_network_shared () {

  char name[256];
  int i_rec, i_cue;
 

  if (T_rec > 0) {
    T_subcycle = N;
    T_rec = fmini(T_rec, T_sim);
  } else if (T_rec < 0) {
    T_subcycle = N / (-T_rec);
    T_rec = 1;
  } else {
    T_subcycle = N;
    T_rec = T_sim;
  }

  if (save_activity) {
    S_bin = malloc(N_bin * sizeof(char));
    sprintf(name, "%s_s.dat", fileroot);
    S_file = fopen(name, "wb");
  }
  sprintf(name, "%s_a.txt", fileroot);
  a_file = fopen(name, "w");
  sprintf(name, "%s_msparse.txt", fileroot);
  overlap_sparse_file = fopen(name, "w");
  sprintf(name, "%s_mcategory.txt", fileroot);
  overlap_cat_file = fopen(name, "w");
  sprintf(name, "%s_mother.txt", fileroot);
  overlap_other_file = fopen(name, "w");


  print_stat("Testing %d cues\n", n_cue);
  q_cue = malloc(n_cue * sizeof(int));

  sample(p*s, fmini(n_cue, p*s), q_cue);
  for (i_cue = p*s; i_cue < n_cue; i_cue++)
    q_cue[i_cue] = q_cue[i_cue%(p*s)];

  output_int_list(q_cue, n_cue, "qcue");

  
  overlap_smean = calloc(T_sim+1, sizeof(double));
  overlap_cmean = calloc(T_sim+1, sizeof(double));

} 


void setup_network_private (unsigned char **S, int **q_update) {

  int t_cycle;

  *S = malloc(N * sizeof(char));

  *q_update = malloc(N * sizeof(int));
  for (t_cycle = 0; t_cycle < N; t_cycle++)
    (*q_update)[t_cycle] = t_cycle;
   
} 

void cleanup_shared () {

  fclose(stat_file);

  if (save_activity)
    fclose(S_file);
  fclose(a_file);
  fclose(overlap_sparse_file);
  fclose(overlap_cat_file);
  fclose(overlap_other_file);

  free(overlap_smean);
  free(overlap_cmean);

}

void cleanup_private (unsigned char *S, int *q_update) {

  free(S);
  free(q_update);

}


//==============================================================================
// END Setup
//==============================================================================



//==============================================================================
// Theta
//==============================================================================

void ramp_theta () {

  int t;
  float tt;

  for (t = 0; t < T_theta; t++) {
    tt = (float)t / T_theta;
    theta[t] = theta_start + (theta_end-theta_start) * tt;
  }
  for (; t < T_sim; t++)
    theta[t] = theta_end;

}

void sine_theta () {

  int t;
  float tt;

  for (t = 0; t < T_theta; t++) {
    tt = (float)t / T_theta;
    theta[t] = theta_end + (theta_start-theta_end)
                                * (cos(2.*M_PI*tt) + 1.) / 2.;
  }
  for (; t < T_sim; t++)
    theta[t] = theta[t%T_theta];

}

void square_theta () {

  int t;

  for (t = 0; t < T_theta; t++)
    theta[t] = theta_start + (theta_end-theta_start) * (t/(T_theta/2));
  for (; t < T_sim; t++)
    theta[t] = theta[t%T_theta];

}

void read_theta () {

  FILE *file;
  int t;

  if ((file = fopen(theta_name, "r")) == NULL) {
    print_err("Error opening theta file %s\n", theta_name);
    exit(1);
  }

  for (t = 0; t < T_sim; t++)
    if (fscanf(file, "%f", &theta[t]) != 1)
      break;

  if (t == 0) {
    print_err("Error reading theta file\n");
    exit(1);
  }

  T_theta = t;
  if (cycle_theta)
    for (; t < T_sim; t++) 
      theta[t] = theta[t%T_theta];
  else
    for (; t < T_sim; t++) 
      theta[t] = theta[T_theta-1];

  fclose(file);

}


void setup_theta () {

  int t;

  theta = calloc(T_sim, sizeof(float));
 
  switch (theta_type) {

    case ramp :
      ramp_theta();
      break;
    case sine :
      sine_theta();
      break;
    case square :
      square_theta();
      break;
    case file :
      read_theta();
      break;

  } 

  print_stat("Theta values: ");
  for (t = 0; t < T_sim; t++)
    print_stat("%3.2f ", theta[t]);
  print_stat("\n");

  output_float_list(theta, T_sim, "theta");

  for (t = 0; t < T_sim; t++)
    theta[t] *= a_fac;

}

//==============================================================================
// END Theta
//==============================================================================



//==============================================================================
// Dynamics
//==============================================================================

void initialize_activity (int i_cue, unsigned char *S) {

  int n_active, n_incomp, n_inacc;
  int *q_active, *q_incomp, *q_inacc;

  int i, k;


  mu_cue = q_cue[i_cue]/s;
  nu_cue = q_cue[i_cue]%s;

  for (i = 0; i < N; i++)
    S[i] = X_sparse[mu_cue][nu_cue][i];

  if (incomp > 0.) {

    n_active = 0;
    q_active = malloc(N * sizeof(int));
    for (i = 0; i < N; i++)
      if (S[i])
        q_active[n_active++] = i;

    n_incomp = lrint(incomp * n_active);
    if (n_incomp == 0)
      print_stat("Warning: no neurons affected by nonzero incomp\n");
    else {
      q_incomp = malloc(n_incomp * sizeof(int));
      sample(n_active, n_incomp, q_incomp);
      for (k = 0; k < n_incomp; k++)
        S[q_active[q_incomp[k]]] = 0;
      free(q_incomp);
    }

    free(q_active);

  }

  if (inacc < 0.) {
    inacc = 1./N;
    print_stat("Initializing cues with single spin flip\n");
  }
  if (inacc > 0.) {

    n_inacc = lrint(inacc * N);
    if (n_inacc == 0)
      print_stat("Warning: no neurons affected by nonzero inacc\n");
    else {
      q_inacc = malloc(n_inacc * sizeof(int));
      sample(N, n_inacc, q_inacc);
      for (k = 0; k < n_inacc; k++)
        S[q_inacc[k]] = 1 - S[q_inacc[k]];
      free(q_inacc);
    }

  }
  
  memcpy(X_cue, S, N*sizeof(char));

} 


void plan_updates (unsigned char *S, int *q_update) {

  int t_cycle;

  switch (update_type) {

    case sequential :
      break;

    case block_random :
      permutation(N, q_update);
      break;

    case full_random :
      for (t_cycle = 0; t_cycle < N; t_cycle++)
        q_update[t_cycle] = (int) ((double)RUNI * N); 
      break;

  }
  
}


void update_activity (unsigned char *S, int q, float theta) {

  float h;

  int j;

  h = 0.;
  for (j = 0; j < N; j++)
    h += w[(long int)q*N+j] * S[j];
  h -= theta;
  h += f * X_cue[q];

  if (beta <= 0.)
    S[q] = step( h );
  else
    S[q] = step( 1./(1. + exp(-beta * h)) - RUNI );

} 


//==============================================================================
// END Dynamics
//==============================================================================



//==============================================================================
// Recording
//==============================================================================


float calculate_sparseness (unsigned char *S) {

  int sum;

  int i;

  sum = 0;

  for (i = 0; i < N; i++)
    sum += S[i];

  return (float) sum / N;

} 

double calculate_overlap (
    unsigned char *S, unsigned char *X, enum pattern_type type
) {

  double sum, dot, overlap;
  double a;

  int i;

  sum = 0.;
  dot = 0.; 
  
  switch (record_type) {
    
    case custom :

      for (i = 0; i < N; i++) {
        sum += X[i];
        dot += S[i] * (2.*X[i] - 1.);
      }
      overlap = fmin(fmax( dot / sum, 0.), 1.);
      break;

    case classic :

      if (type == sparse)
        a = a_sparse;
      else
        a = a_dense;
      
      for (i = 0; i < N; i++)
        dot += S[i] * (X[i]-a);

      overlap = dot / (N * a*(1.-a));
      break;

  }

  return overlap;

}


void record_subcycle (unsigned char *S) {

  double overlap;
  int mu, nu;

  
  if (save_activity) {
    compress_binary_char(S, S_bin, N);
    fwrite(S_bin, sizeof(char), N_bin, S_file);
  }

  fprintf(a_file, "%.2e ", calculate_sparseness(S));

  overlap = calculate_overlap(S, X_sparse[mu_cue][nu_cue], sparse);
  fprintf(overlap_sparse_file, "% 4.3f ", overlap);

  for (nu = 0; nu < s; nu++) {
    if (nu == nu_cue)
      continue;
    overlap = calculate_overlap(S, X_sparse[mu_cue][nu], sparse);
    fprintf(overlap_sparse_file, "% 4.3f ", overlap);
  }
  fprintf(overlap_sparse_file, "\n");

  for (mu = 0; mu < p; mu++) {
    if (mu == mu_cue)
      continue;
    for (nu = 0; nu < s; nu++) {
      overlap = calculate_overlap(S, X_sparse[mu][nu], sparse);
      fprintf(overlap_other_file, "% 4.3f ", overlap);
    }
  }
  fprintf(overlap_other_file, "\n");

  overlap = calculate_overlap(S, X_cat[mu_cue], cat);
  fprintf(overlap_cat_file, "% 4.3f ", overlap);

  for (mu = 0; mu < p; mu++) {
    if (mu == mu_cue)
      continue;
    overlap = calculate_overlap(S, X_cat[mu], cat);
    fprintf(overlap_cat_file, "% 4.3f ", overlap);
  }
  fprintf(overlap_cat_file, "\n");

}


void record_cycle (int t, unsigned char *S) {

  double overlap;

  
  overlap = calculate_overlap(S, X_sparse[mu_cue][nu_cue], sparse);
  overlap_smean[t] += fabs(overlap) / n_cue;

  overlap = calculate_overlap(S, X_cat[mu_cue], cat);
  overlap_cmean[t] += fabs(overlap) / n_cue;

  if (t == T_sim) {
    fprintf(a_file, "\n");
    fprintf(overlap_sparse_file, "\n");
    fprintf(overlap_cat_file, "\n");
    fprintf(overlap_other_file, "\n");
  }

}

//==============================================================================
// END Recording
//==============================================================================




int main (int argc, char *argv[]) {

  unsigned char *S;
  int *q_update;

  clock_t tic, toc;

  int i_cue, t, t_cycle;


  setup_parameters(argc, argv);

  print_stat("\nSTARTING NETWORK SETUP\n");

  setup_patterns();

  setup_theta();

  setup_network_shared();


  
  print_stat("\nSTARTING SIMULATION\n");

  tic = clock();
  setup_network_private(&S, &q_update);

  for (i_cue = 0; i_cue < n_cue; i_cue++) {

    print_screen("%2d/%2d cues tested...\r", i_cue, n_cue);

    initialize_activity(i_cue, S);
    record_subcycle(S);
    record_cycle(0, S);
    
    for (t = 1; t <= T_sim; t++) {

      plan_updates(S, q_update);
      for (t_cycle = 1; t_cycle <= N; t_cycle++) {

        update_activity(S, q_update[t_cycle-1], theta[t-1]);
        if (t % T_rec == 0 && t_cycle % T_subcycle == 0)
          record_subcycle(S);

      }

      record_cycle(t, S);

    }

  }

  clear_screen();


  cleanup_private(S, q_update);
  toc = clock();
  print_stat("Done in %.2f s\n", (double) (toc-tic)/CLOCKS_PER_SEC);

  print_stat("Mean sparse   overlaps: ");
  for (t = 0; t <= T_sim; t++)
    print_stat("%4.3f ", overlap_smean[t]);
  print_stat("\n");
  print_stat("Mean category overlaps: ");
  for (t = 0; t <= T_sim; t++)
    print_stat("%4.3f ", overlap_cmean[t]);
  print_stat("\n");

  print_stat("\nSIMULATION COMPLETE\n\n");

  cleanup_shared();

}
