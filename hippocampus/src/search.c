/*******************************************************************************
 *
 *  FILE:         search.c
 *
 *  DATE:         24 February 2023
 *
 *  AUTHOR:       Louis Kang
 *
 *  LICENSE:      GPLv3
 *
 *  REFERENCE:    To be determined
 *
 *  PURPOSE:      Simulating a Hopfield model with dual encodings. A search over
 *                the inhibition level can be performed to maximize overlap.
 *
 *  DEPENDENCIES: Intel Math Kernel Library
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

// See Reading in parameters section for the syntax for setting parameters as
// command line arguments.

// Random ----------------------------------------------------------------------
uint32_t r4seed = 0;      // 0: use time as random seed
#define RUNI r4_uni(&r4seed)

// Iterations and time ---------------------------------------------------------
int T_sim = 100;    // Main simulation timesteps
int T_rec = 10;     // Print every tscreen timesteps
enum sequence_type {sequential, block_random, full_random};
enum sequence_type update_type = block_random;

// Network ---------------------------------------------------------------------
int N;                // number of neurons
int p = 1;            // number of categories stored
int s = 10;           // number of examples per category stored
int N_bin;
int p_max;            // number of categories in file
int s_max;            // number of examples per category in file
float g = 0.1;        // dense pattern strength; half of the corresponding value
                      //   in the manuscript.
float beta = 0.;      // rescaled inverse temperature
float a_sparse;       // sparse pattern sparsity
float a_dense;        // dense pattern sparsity
float a_fac;

float *w;

// Pattern ---------------------------------------------------------------------
char X_dir[256] = "";
int n_cue = 100;
enum pattern_type {sparse, dense, cat, both};
enum pattern_type cue_type = sparse;
enum pattern_type target_type = sparse;
float incomp = 0.;   // fraction of active neurons to inactivate in cue
float inacc = 0.;    // fraction of random flips in cue

int sparse_file, dense_file, cat_file;
int shuffle_patterns = 1;
int *q_mu, *q_nu;
unsigned char **X_sparse, **X_dense, **X_cat;
unsigned char **X_cue, **X_target;
unsigned char *X_bin;

// Recording -------------------------------------------------------------------
FILE *S_file, *a_file, *overlap_custom_file, *overlap_classic_file;
unsigned char *S_bin;
double overlap_start, overlap_end;

// Theta search ----------------------------------------------------------------
int search_over_theta = 1;
float theta_mid = 0.;
int n_round = 2;
int n_value = 4;

float theta_sim;
float theta_max;
double overlap_max;
float *theta_search;
double *overlap_search;
enum overlap_type {classic, custom};
enum overlap_type criterion_type = classic;

int T_search = 3;
int n_search = 20;

// I/O -------------------------------------------------------------------------
char fileroot[256] = "";  // Output filename root
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


// Circularly rotate byte c by n places leftward
static inline unsigned char rotl (unsigned char c, unsigned int n) {

  n &= 7;
  return (c << n) | (c >> ((-n) & 7));

}

// Circularly rotate byte c by n places rightward
static inline unsigned char rotr (unsigned char c, unsigned int n) {

  n &= 7;
  return (c >> n) | (c << ((-n) & 7));

}

// Implements np.packbits with bitorder='big'
void compress_binary_char (
    unsigned char *raw, unsigned char *bin, int n_raw
) {

  int i;

  for (i = 0; i < n_raw; i += 8)
    bin[i/8] = 0;

  for (i = 0; i < n_raw; i++)
    bin[i/8] |= raw[i] << (7 - (i % 8));

}

// Implements np.unpackbits with bitorder='big'
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

// Generate random real between 0 and 1 based on the SHR3 Xorshift random number
// generator developed by George Marsaglia and implemented by John Burkhardt.
// See <https://people.sc.fsu.edu/~jburkardt/c_src/ziggurat/ziggurat.html>.
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

void permutation (int n, int *arr) {
  
  int i, j;

  for (i = 0; i < n; i++)
    arr[i] = i;

  for (i = n-1; i > 0; i--) {
    j = (int) ((double)RUNI * (i+1));
    swap(&arr[i], &arr[j]);
  }

}

// Algorithm for sampling without replacement
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
    }

    else if (!strcmp(argv[narg],"-X_dir")) {
      sscanf(argv[narg+1],"%s",X_dir);
      narg += 2;
    } else if (!strcmp(argv[narg],"-no_shuffle")) {
      shuffle_patterns = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-n_cue")) {
      sscanf(argv[narg+1],"%d",&n_cue);
      narg += 2;
    } else if (!strcmp(argv[narg],"-sparse_cue")) {
      cue_type = sparse;
      narg++;
    } else if (!strcmp(argv[narg],"-dense_cue")) {
      cue_type = dense;
      narg++;
    } else if (!strcmp(argv[narg],"-category_cue")) {
      cue_type = cat;
      narg++;
    } else if (!strcmp(argv[narg],"-both_cue")) {
      cue_type = both;
      narg++;
    } else if (!strcmp(argv[narg],"-sparse_target")) {
      target_type = sparse;
      narg++;
    } else if (!strcmp(argv[narg],"-dense_target")) {
      target_type = dense;
      narg++;
    } else if (!strcmp(argv[narg],"-category_target")) {
      target_type = cat;
      narg++;
    } else if (!strcmp(argv[narg],"-incomp")) {
      sscanf(argv[narg+1],"%f",&incomp);
      narg += 2;
    } else if (!strcmp(argv[narg],"-inacc")) {
      sscanf(argv[narg+1],"%f",&inacc);
      narg += 2;
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
    }

    else if (!strcmp(argv[narg],"-search")) {
      search_over_theta = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-no_search")) {
      search_over_theta = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-custom_overlap")) {
      criterion_type = custom;
      narg++;
    } else if (!strcmp(argv[narg],"-classic_overlap")) {
      criterion_type = classic;
      narg++;
    } else if (!strcmp(argv[narg],"-theta")) {
      sscanf(argv[narg+1],"%f",&theta_mid);
      search_over_theta = 0;
      narg += 2;
    } else if (!strcmp(argv[narg],"-theta_mid")) {
      sscanf(argv[narg+1],"%f",&theta_mid);
      narg += 2;
    } else if (!strcmp(argv[narg],"-n_round")) {
      sscanf(argv[narg+1],"%d",&n_round);
      narg += 2;
    } else if (!strcmp(argv[narg],"-n_value")) {
      sscanf(argv[narg+1],"%d",&n_value);
      narg += 2;
    } else if (!strcmp(argv[narg],"-T_search")) {
      sscanf(argv[narg+1],"%d",&T_search);
      narg += 2;
    } else if (!strcmp(argv[narg],"-n_search")) {
      sscanf(argv[narg+1],"%d",&n_search);
      narg += 2;
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

  fprintf(file, "seed             = %d\n", r4seed);

  fprintf(file, "\n");
  fprintf(file, "T_sim            = %d\n", T_sim);
  fprintf(file, "T_rec            = %d\n", T_rec);
  fprintf(file, "update_type      = %d\n", update_type);

  fprintf(file, "\n");
  fprintf(file, "N                = %d\n", N);
  fprintf(file, "p                = %d\n", p);
  fprintf(file, "s                = %d\n", s);
  fprintf(file, "gamma            = %f\n", g);
  fprintf(file, "beta             = %f\n", beta);

  fprintf(file, "\n");
  fprintf(file, "X_dir            = %s\n", X_dir);
  fprintf(file, "shuffle_patterns = %d\n", shuffle_patterns);
  fprintf(file, "n_cue            = %d\n", n_cue);
  fprintf(file, "cue_type         = %d\n", cue_type);
  fprintf(file, "target_type      = %d\n", target_type);
  fprintf(file, "incomp           = %f\n", incomp);
  fprintf(file, "inacc            = %f\n", inacc);

  fprintf(file, "\n");
  fprintf(file, "search           = %d\n", search_over_theta);
  fprintf(file, "criterion_type   = %d\n", criterion_type);
  fprintf(file, "theta_mid        = %f\n", theta_mid);
  fprintf(file, "n_round          = %d\n", n_round);
  fprintf(file, "n_value          = %d\n", n_value);
  fprintf(file, "T_search         = %d\n", T_search);
  fprintf(file, "n_search         = %d\n", n_search);

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
  
  // Pattern directory must contain pattern statistics in a strict format
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
    print_stat("Sparsenesses detected: %4.3f and %4.3f\n", a_sparse, a_dense);
  }


  // Setting random number seed using time
  if (r4seed == 0) {
    r4seed = (uint32_t)(time(NULL) % 1048576);
    place = 1;
    for (ch = 0; ch < strlen(fileroot); ch++) {
      r4seed += fileroot[ch] * place;
      place = (place * 128) % 100000;
    }
  }

  // Number of packed bytes that contains pattern of length N
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

  
  // q_mu contains indices of stored concepts
  q_mu = malloc(p * sizeof(int));
  if (shuffle_patterns)
    sample(p_max, p, q_mu);
  else
    for (mu = 0; mu < p; mu++)
      q_mu[mu] = mu;
  output_int_list(q_mu, p, "qmu");

  // q_mu contains indices of stored examples in each concept
  q_nu = malloc(s * sizeof(int));
  if (shuffle_patterns)
    sample(s_max, s, q_nu);
  else
    for (nu = 0; nu < s; nu++)
      q_nu[nu] = nu;
  output_int_list(q_nu, s, "qnu");


  // X_sparse and X_dense contain sparse and dense examples
  X_bin = malloc(N_bin * sizeof(char *));
  X_sparse = malloc(s * sizeof(char *));
  for (nu = 0; nu < s; nu++)
    X_sparse[nu] = malloc(N * sizeof(char));
  X_dense = malloc(s * sizeof(char *));
  for (nu = 0; nu < s; nu++)
    X_dense[nu] = malloc(N * sizeof(char));

  // w is the (flattened) connectivity matrix
  X_combo = mkl_malloc(s*N * sizeof(float), 64);
  w = mkl_calloc(N*N, sizeof(float *), 64);


  for (mu = 0; mu < p; mu++) {

    print_screen("%3d/%3d categories stored...\r", mu, p);

    for (nu = 0; nu < s; nu++) {

      read_pattern(X_sparse[nu], sparse_file, mu, nu);
      read_pattern(X_dense [nu], dense_file , mu, nu);

    }

    // Note that 2*g is the dense strength zeta in the manuscript
    for (nu = 0; nu < s; nu++)
      for (i = 0; i < N; i++)
        X_combo[nu*N+i] =   (1.-2.*g) * (X_sparse[nu][i] - a_sparse)
                          + (   2.*g) * (X_dense [nu][i] - a_dense );

    // Add X_combo multiplied with its transpose to w
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

  for (nu = 0; nu < s; nu++)
    free(X_sparse[nu]);
  free(X_sparse);
  for (nu = 0; nu < s; nu++)
    free(X_dense[nu]);
  free(X_dense);
  mkl_free(X_combo);

  
  // Theoretically motiviated rescaling of beta
  a_fac = (1.-2.*g)*(1.-2.*g) * a_sparse;
  beta /= a_fac;

}


void setup_cue_target (unsigned char **X, int *q_cue, enum pattern_type type) {

  unsigned char *X_buf;
  int nu_cue, i;

  switch (type) {

    case sparse :
      for (nu_cue = 0; nu_cue < n_cue; nu_cue++)
        read_pattern(X[nu_cue], sparse_file, q_cue[nu_cue]/s, q_cue[nu_cue]%s);
      break;

    case dense :
      for (nu_cue = 0; nu_cue < n_cue; nu_cue++)
        read_pattern(X[nu_cue], dense_file, q_cue[nu_cue]/s, q_cue[nu_cue]%s);
      break;

    case cat :
      for (nu_cue = 0; nu_cue < n_cue; nu_cue++)
        read_pattern(X[nu_cue], cat_file, q_cue[nu_cue]/s, -1);
      break;

    case both :
      X_buf = malloc(N * sizeof(char));
      for (nu_cue = 0; nu_cue < n_cue; nu_cue++) {
        read_pattern(X[nu_cue], sparse_file, q_cue[nu_cue]/s, q_cue[nu_cue]%s);
        read_pattern(X_buf,     dense_file,  q_cue[nu_cue]/s, q_cue[nu_cue]%s);
        for (i = 0; i < N; i++)
          X[nu_cue][i] |= X_buf[i];
      }
      break;

  }
  
}


// Variables shared among cues
void setup_network_shared () {

  int *q_cue;

  char name[256];
  int i_rec, nu_cue;
 

  // Record data every T_rec update cycles
  T_rec = fmini(T_rec, T_sim);
  overlap_start = 0.;
  overlap_end = 0.;

  // Save raw network activities as packed binary file
  if (save_activity) {
    S_bin = malloc(N_bin * sizeof(char));
    sprintf(name, "%s_s.dat", fileroot);
    S_file = fopen(name, "wb");
  }
  sprintf(name, "%s_a.txt", fileroot);
  a_file = fopen(name, "w");
  sprintf(name, "%s_mcustom.txt", fileroot);
  overlap_custom_file = fopen(name, "w");
  sprintf(name, "%s_mclassic.txt", fileroot);
  overlap_classic_file = fopen(name, "w");


  print_stat("Testing %d cues\n", n_cue);
  q_cue = malloc(n_cue * sizeof(int));

  if (cue_type == cat && target_type == cat) {

    if (shuffle_patterns)
      sample(p, fmini(n_cue, p), q_cue);
    else
      for (nu_cue = 0; nu_cue < fmini(n_cue, p); nu_cue++)
        q_cue[nu_cue] = nu_cue;

    // Cycle through patterns if fewer than n_cue are stored
    for (nu_cue = p; nu_cue < n_cue; nu_cue++)
      q_cue[nu_cue] = q_cue[nu_cue%p];

    // Sets q_cue to first example in each selected concept; example identity
    // doesn't matter since we are only interested in cat cues and targets
    for (nu_cue = 0; nu_cue < n_cue; nu_cue++)
      q_cue[nu_cue] *= s;

  } else {

    if (shuffle_patterns)
      sample(p*s, fmini(n_cue, p*s), q_cue);
    else
      for (nu_cue = 0; nu_cue < fmini(n_cue, p*s); nu_cue++)
        q_cue[nu_cue] = (nu_cue%p) * s + nu_cue/p;

    for (nu_cue = p*s; nu_cue < n_cue; nu_cue++)
      q_cue[nu_cue] = q_cue[nu_cue%(p*s)];

  }

  output_int_list(q_cue, n_cue, "qcue");


  X_cue = malloc(n_cue * sizeof(char *));
  for (nu_cue = 0; nu_cue < n_cue; nu_cue++)
    X_cue[nu_cue] = malloc(N * sizeof(char));

  X_target = malloc(n_cue * sizeof(char *));
  for (nu_cue = 0; nu_cue < n_cue; nu_cue++)
    X_target[nu_cue] = malloc(N * sizeof(char));


  setup_cue_target(X_cue   , q_cue, cue_type   );
  setup_cue_target(X_target, q_cue, target_type);

  free(q_cue);
   
} 

// Variables private to each cue; can potentially be parallelized, but not
// implemented here
void setup_network_private (
    unsigned char **S, int **q_update, float *theta, float theta_base
) {

  int t_cycle;

  *S = malloc(N * sizeof(char));

  *q_update = malloc(N * sizeof(int));
  for (t_cycle = 0; t_cycle < N; t_cycle++)
    (*q_update)[t_cycle] = t_cycle;

  *theta = theta_base * a_fac;
   
} 

void cleanup_shared () {

  fclose(stat_file);
  close(sparse_file);
  close(dense_file);
  close(cat_file);

  if (save_activity)
    fclose(S_file);
  fclose(a_file);
  fclose(overlap_custom_file);
  fclose(overlap_classic_file);

}

void cleanup_private (unsigned char *S, int *q_update) {

  free(S);
  free(q_update);

}

//==============================================================================
// END Setup
//==============================================================================



//==============================================================================
// Dynamics
//==============================================================================

void initialize_activity (int nu_cue, unsigned char *S) {

  int n_active, n_incomp, n_inacc;
  int *q_active, *q_incomp, *q_inacc;

  int i, k;

  // Initialize network to cue
  for (i = 0; i < N; i++)
    S[i] = X_cue[nu_cue][i];

  // Randomly deactivate neurons
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

  // Randomly flip neurons between active and inactive
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

} 


void plan_updates (unsigned char *S, int *q_update) {

  int t_cycle;

  switch (update_type) {

    // update neurons in sequence
    case sequential :
      break;

    // update one neuron per update cycle in random order
    case block_random :
      permutation(N, q_update);
      break;

    // update neurons randomly
    case full_random :
      for (t_cycle = 0; t_cycle < N; t_cycle++)
        q_update[t_cycle] = (int) ((double)RUNI * N); 
      break;

  }
  
}


// Glauber dynamics
void update_activity (unsigned char *S, int q, float theta) {

  float h;

  int j;

  h = 0.;
  for (j = 0; j < N; j++)
    h += w[(long int)q*N+j] * S[j];
  h -= theta;
  
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

// Only classic overlap was used in the manuscript
double calculate_overlap (
    unsigned char *S, unsigned char *X, enum overlap_type type
) {

  double sum, dot, overlap;
  double a;

  int i;

  sum = 0.;
  dot = 0.; 
  
  switch (type) {
    
    case custom :

      for (i = 0; i < N; i++) {
        sum += X[i];
        dot += S[i] * (2.*X[i] - 1.);
      }
      overlap = fmin(fmax( dot / sum, 0.), 1.);
      break;

    case classic :

      if (target_type == sparse)
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

void record_activity (int t, int nu_cue, unsigned char *S) {

  double overlap_custom, overlap_classic;

  if (save_activity) {
    compress_binary_char(S, S_bin, N);
    fwrite(S_bin, sizeof(char), N_bin, S_file);
  }

  fprintf(a_file, "%.2e ", calculate_sparseness(S));

  overlap_custom  = calculate_overlap(S, X_target[nu_cue], custom );
  overlap_classic = calculate_overlap(S, X_target[nu_cue], classic);

  fprintf(overlap_custom_file , "% 4.3f ", overlap_custom );
  fprintf(overlap_classic_file, "% 4.3f ", overlap_classic);

  if (t == 0)
    overlap_start +=
        fabs(calculate_overlap(S, X_target[nu_cue], criterion_type))
            / n_cue;
  if (t == T_sim)
    overlap_end +=
        fabs(calculate_overlap(S, X_target[nu_cue], criterion_type))
            / n_cue;

  if (t == T_sim) {
    fprintf(a_file, "\n");
    fprintf(overlap_custom_file , "\n");
    fprintf(overlap_classic_file, "\n");
  }

}

//==============================================================================
// END Recording
//==============================================================================



//==============================================================================
// Search over theta
//==============================================================================

// These functions allow the network to iteratively search over theta values to
// optimize overlap with the target pattern to a resolution of 0.01 in theta.
void plan_search () {

  float theta_width;

  int i_round, i_value;

  n_search = fmini(n_search, n_cue);
  T_search = fmini(T_search, T_sim);

  // In each search round, n_value theta values are searched. The value that
  // maximizes overlap with the target pattern is used as the midpoint for the
  // next round. The range of theta values searched spans from
  // theta_mid-theta_width to theta_mid+theta_width.
  theta_width = 0.;
  for (i_round = 0; i_round < n_round; i_round++)
    theta_width += 0.01 * pow(n_value-1., i_round+1) / 2.;

  print_stat("Maximizing overlap w.r.t. theta using %d rounds of %d values\n",
             n_round, n_value);
  print_stat("Theta search range from %4.3f to %4.3f\n",
             theta_mid-theta_width, theta_mid+theta_width);

  // Initial (coarsest) search values
  theta_search = malloc(n_value * sizeof(float));
  for (i_value = 0; i_value < n_value; i_value++)
    theta_search[i_value] = theta_mid + 0.01 * pow(n_value-1, n_round-1) *
                                        (-(n_value-1.)/2. + i_value);

  overlap_search = malloc(n_value * sizeof(double));

}


int update_search (int i_round) {

  int q_max1, q_max2;
  int rounds_left;
  
  int i_value;

  // Find argmax of overlap. Allow for tie between q_max1 and q_max2.
  overlap_max = overlap_search[0];
  q_max1 = 0;
  q_max2 = 0;
  for (i_value = 1; i_value < n_value; i_value++) {

    if (overlap_search[i_value] > overlap_max) {
      overlap_max = overlap_search[i_value];
      q_max1 = i_value;
      q_max2 = i_value;
    } else if (overlap_search[i_value] == overlap_max)
      q_max2 = i_value;

  }

  if (q_max1 != q_max2) {
    print_stat(
        "Maximum overlap achieved by multiple theta values, reporting mean\n"
    );
    theta_max = (theta_search[q_max1] + theta_search[q_max2]) / 2.;
  } else
    theta_max = theta_search[q_max1];

  // Stop search if highest overlap is achieved
  if (overlap_max > 0.9995) {
    print_stat("Full overlap of 1.000 achieved by theta = %4.3f in round %d\n",
               theta_max, i_round+1);
    return 1;
  } else {
    print_stat("Mean overlap of %4.3f achieved by theta = %4.3f in round %d\n",
               overlap_max, theta_max, i_round+1);

    // Update theta_search for the next round with finer values centered around
    // the best value discovered this round
    rounds_left = n_round - (i_round+1);
    if (rounds_left > 0)
      for (i_value = 0; i_value < n_value; i_value++)
        theta_search[i_value] = theta_max +
                                0.01 * pow(n_value-1, rounds_left-1) *
                                (-(n_value-1.)/2. + i_value);

    return 0;
  }

}

void print_theta () {

  FILE *file;
  char name[256];

  sprintf(name, "%s_params.txt", fileroot);
  file = fopen(name, "a");

  // Save theta value that maximized overlap
  fprintf(file, "\n");
  fprintf(file, "theta_max    = %f\n", theta_max);

  fclose(file);

}

//==============================================================================
// END Search over theta
//==============================================================================




int main (int argc, char *argv[]) {

  unsigned char *S;
  int *q_update;
  float theta;

  clock_t tic, toc;

  int i_round, i_value;
  int nu_cue, t, t_cycle;


  setup_parameters(argc, argv);

  print_stat("\nSTARTING NETWORK SETUP\n");

  setup_patterns();

  setup_network_shared();


  // Search over theta to maximize mean overlap with target patterns
  if (!search_over_theta) 
    theta_sim = theta_mid;
  else  {

    plan_search();

    for (i_round = 0; i_round < n_round; i_round++) {

      print_stat("\nSTARTING THETA SEARCH ROUND %d of %d\n\n",
                 i_round+1, n_round);

      for (i_value = 0; i_value < n_value; i_value++) {

        tic = clock();
        setup_network_private(&S, &q_update, &theta, theta_search[i_value]);
        overlap_search[i_value] = 0.;
        
        for (nu_cue = 0; nu_cue < n_search; nu_cue++) {

          initialize_activity(nu_cue, S);
          for (t = 1; t <= T_search; t++) {
            plan_updates(S, q_update);
            for (t_cycle = 0; t_cycle < N; t_cycle++) 
              update_activity(S, q_update[t_cycle], theta);
          }

          overlap_search[i_value] +=
              fabs(calculate_overlap(S, X_target[nu_cue], criterion_type))
                  / n_search;

        }

        cleanup_private(S, q_update);
        toc = clock();
        print_stat(
            "Round %d, theta = %4.3f: done in %.2f s, mean overlap %4.3f\n",
            i_round+1, theta_search[i_value], (double) (toc-tic)/CLOCKS_PER_SEC,
            overlap_search[i_value]
        );

      }

      if (update_search(i_round))
        break;

    }
    
    print_stat("\nTHETA SEARCH: %4.3f ACHIEVES MAXIMUM MEAN OVERLAP %4.3f\n",
                theta_max, overlap_max);
    print_theta();
    theta_sim = theta_max;

  }



  // Perform final simulation at optimal theta and record data
  print_stat("\nSTARTING FINAL SIMULATION\n");

  tic = clock();
  setup_network_private(&S, &q_update, &theta, theta_sim);

  for (nu_cue = 0; nu_cue < n_cue; nu_cue++) {

    print_screen("%2d/%2d cues tested...\r", nu_cue, n_cue);

    initialize_activity(nu_cue, S);
    record_activity(0, nu_cue, S);
    
    for (t = 1; t <= T_sim; t++) {

      plan_updates(S, q_update);
      for (t_cycle = 0; t_cycle < N; t_cycle++) 
        update_activity(S, q_update[t_cycle], theta);
      if (t % T_rec == 0)
        record_activity(t, nu_cue, S);

    }

  }

  clear_screen();


  cleanup_private(S, q_update);
  toc = clock();
  print_stat("Done in %.2f s\n", (double) (toc-tic)/CLOCKS_PER_SEC);

  print_stat("Mean overlap %4.3f --> %4.3f", overlap_start, overlap_end);
  if (search_over_theta)
    print_stat(" achieved by theta = %4.3f", theta_sim);
  print_stat("\n");

  print_stat("\nSIMULATION COMPLETE\n\n");

  cleanup_shared();

}
