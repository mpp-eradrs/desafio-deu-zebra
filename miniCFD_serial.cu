//////////////////////////////////////////////////////////////////////////////////////////
// miniCFD
// Author: Omitted
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <ctime>
#include <iostream>

const double pi        = 3.14159265358979323846264338327;   //Pi
const double grav      = 9.8;                               //Gravitational acceleration (m / s^2)
const double cp        = 1004.;                             //Specific heat of dry air at constant pressure
const double cv        = 717.;                              //Specific heat of dry air at constant volume
const double rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
const double p0        = 1.e5;                              //Standard pressure at the surface in Pascals
const double C0        = 27.5629410929725927310572984382;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
const double gamm      = 1.40027894002789401278940017893;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
const double xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
const double zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
const double hv   = 0.25;     //How strong to diffuse the solution: hv \in [0:1]
const double cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
const double max_speed = 450;        //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
const int cfd_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
const int NUM_VARS = 4;           //Number of fluid state variables
const int POS_DENS  = 0;           //index for density ("rho")
const int POS_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
const int POS_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
const int POS_RHOT  = 3;           //index for density * potential temperature ("rho * theta")
const int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
const int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction
enum test_cases {CONFIG_IN_TEST1, CONFIG_IN_TEST2, CONFIG_IN_TEST3,
    CONFIG_IN_TEST4, CONFIG_IN_TEST5, CONFIG_IN_TEST6 };

const int nqpoints = 3;
double qpoints [] = { 0.112701665379258311482074460012E0 , 0.510000000000000000000000000000E0 , 0.887298334621741688517926529880E0 };
double qweights[] = { 0.277777777777777777777777777778E0 , 0.444444444444444444444444444445E0 , 0.277777777777777777777777777786E0 };

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
const double sim_time = _SIM_TIME;              //total simulation time in seconds
const double output_freq = _OUT_FREQ;           //frequency to perform output in seconds
double dt;                    //Model time step (seconds)
const int    nx_cfd = _NX, nz_cfd = _NZ;      //Number of total grid cells in the x- and z- dimensions
const int    nnx = nx_cfd, nnz = nz_cfd;                //Number of local grid cells in the x- and z- dimensions for this MPI task
const double dx = xlen / nx_cfd, dz = zlen / nz_cfd;;                //Grid space length in x- and z-dimension (meters)
const int    i_beg = 0, k_beg = 0;          //beginning index in the x- and z-directions for this MPI task
const int    nranks = 1, myrank = 0;        //Number of MPI ranks and my rank id
const int    masterproc = (myrank == 0);            //Am I the master process (rank == 0)?
const int    config_spec = _IN_CONFIG;         //Which data initialization to use
double *cfd_dens_cell_cpu;         //density (vert cell avgs).   Dimensions: (1-hs:nnz+hs)
double *cfd_dens_cell_gpu;
double *cfd_dens_theta_cell_cpu;   //rho*t (vert cell avgs).     Dimensions: (1-hs:nnz+hs)
double *cfd_dens_theta_cell_gpu;
double *cfd_dens_int_cpu;          //density (vert cell interf). Dimensions: (1:nnz+1)
double *cfd_dens_int_gpu;
double *cfd_dens_theta_int_cpu;    //rho*t (vert cell interf).   Dimensions: (1:nnz+1)
double *cfd_dens_theta_int_gpu;
double *cfd_pressure_int_cpu;      //press (vert cell interf).   Dimensions: (1:nnz+1)
double *cfd_pressure_int_gpu;

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double etime;                 //Elapsed model time
double output_counter;        //Helps determine when it's time to do output
//Runtime variable arrays
double *state_cpu;                //Fluid state.             Dimensions: (1-hs:nnx+hs,1-hs:nnz+hs,NUM_VARS)
double *state_gpu;
double *state_tmp_cpu;            //Fluid state.             Dimensions: (1-hs:nnx+hs,1-hs:nnz+hs,NUM_VARS)
double *state_tmp_gpu;
double *flux_cpu;                 //Cell interface fluxes.   Dimensions: (nnx+1,nnz+1,NUM_VARS)
double *flux_gpu;
double *tend_cpu;                 //Fluid state tendencies.  Dimensions: (nnx,nnz,NUM_VARS)
double *tend_gpu;
int    num_out = 0;           //The number of outputs performed so far
int    direction_switch = 1;
double mass0, te0;            //Initial domain totals for mass and total energy  
double mass , te ;            //Domain totals for mass and total energy  

//How is this not in the standard?!
double dmin( double a , double b ) { if (a<b) {return a;} else {return b;} };


//Declaring the functions defined after "main"
void   initialize                 ( int *argc , char ***argv );
void   finalize             ( );
void   testcase6            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase5      ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase4           ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase3       ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase2              ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase1            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   const_theta    ( double z                   , double &r , double &t );
void   const_bvfreq   ( double z , double bv_freq0 , double &r , double &t );

double sample_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad );
void   output               ( double *state , double etime );
void   do_timestep     ( double *state , double *state_tmp , double *flux , double *tend , double dt );
void   do_semi_step   ( double *state_init , double *state_forcing , double *state_out , double dt , int dir , double *flux , double *tend );
void   do_dir_x ( double *state , double *flux , double *tend );
void   do_dir_z ( double *state , double *flux , double *tend );
void   exchange_border_x    ( double *state );
void   exchange_border_z    ( double *state );
void   do_results           ( double &mass , double &te );

const int blockSize = 1024;

__global__ void do_semi_step_add(double *state_out, double *state_init, double *tend, int n, int dt){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	if(id < n){
		  int ll= id /(nnz * nnx);
    	int k = (id / nnx) % nnz;
    	int i = id % nnx;
		  int inds = (k+hs)*(nnx+2*hs) + ll*(nnz+2*hs)*(nnx+2*hs) + i+hs;
    	int indt = ll*nnz*nnx + k*nnx + i;
    	state_out[inds] = state_init[inds] + dt * tend[indt];
	}
}

__global__ void do_dir_x_flux(double *state, double *flux, double *tend, double *cfd_dens_cell, double *cfd_dens_theta_cell, int n, double v_coef){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	if(id < n){
		
		int k = id / (nnx + 1);
    int i = id % (nnx + 1);

		double vals[NUM_VARS], d_vals[NUM_VARS];

		for (int ll=0; ll<NUM_VARS; ll++) {
      
			double stencil[4];
		
			for (int s=0; s < cfd_size; s++) {
				int inds = ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+s;
				stencil[s] = state[inds];
			}
		
			//Fourth-order-accurate interpolation of the state
			vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
			//First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
			d_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
		}

		double r = vals[POS_DENS] + cfd_dens_cell[k+hs];
		double u = vals[POS_UMOM] / r;
		double w = vals[POS_WMOM] / r;
		double t = ( cfd_dens_theta_cell[k+hs] + vals[POS_RHOT] ) / r;
		double p = pow((r*t),gamm)*C0;

    flux[POS_DENS*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u     - v_coef*d_vals[POS_DENS];
		flux[POS_UMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*u+p - v_coef*d_vals[POS_UMOM];
		flux[POS_WMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*w   - v_coef*d_vals[POS_WMOM];
		flux[POS_RHOT*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*t   - v_coef*d_vals[POS_RHOT];

  }
}

__global__ void do_dir_x_add(double *tend, double *flux, int n){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	if(id < n){
		int ll= id /(nnz * nnx);
    	int k = (id / nnx) % nnz;
    	int i = id % nnx;
		int indt  = ll* nnz   * nnx    + k* nnx    + i  ;
		int indf1 = ll*(nnz+1)*(nnx+1) + k*(nnx+1) + i  ;
		int indf2 = ll*(nnz+1)*(nnx+1) + k*(nnx+1) + i+1;
		tend[indt] = -( flux[indf2] - flux[indf1] ) / dx;
	}
}

__global__ void do_dir_z_flux(double *state , double *flux, double *tend, double *cfd_dens_int, double *cfd_dens_theta_int, double *cfd_pressure_int, int n, double v_coef){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;

  if(id == 0){
    //printf("kernel_z_flux\n");
  }
	
	if(id < n){
	
		int k = id / nnx;
    	int i = id % nnx;
		//Use fourth-order interpolation from four cell averages to compute the value at the interface in question
		
		double stencil[4], d_vals[NUM_VARS], vals[NUM_VARS];
		
		for (int ll=0; ll<NUM_VARS; ll++) {
			for (int s=0; s<cfd_size; s++) {
				int inds = ll*(nnz+2*hs)*(nnx+2*hs) + (k+s)*(nnx+2*hs) + i+hs;
				stencil[s] = state[inds];
			}
			//Fourth-order-accurate interpolation of the state
			vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
			//First-order-accurate interpolation of the third spatial derivative of the state
			d_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
		}

		//Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
		double r = vals[POS_DENS] + cfd_dens_int[k];
		double u = vals[POS_UMOM] / r;
		double w = vals[POS_WMOM] / r;
		double t = ( vals[POS_RHOT] + cfd_dens_theta_int[k] ) / r;
		double p = C0*pow((r*t),gamm) - cfd_pressure_int[k];
		
    //Enforce vertical boundary condition and exact mass conservation
		if (k == 0 || k == nnz) {
			w = 0;
			d_vals[POS_DENS] = 0;
		}

		//Compute the flux vector with viscosity
		flux[POS_DENS*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w     - v_coef*d_vals[POS_DENS];
		flux[POS_UMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*u   - v_coef*d_vals[POS_UMOM];
		flux[POS_WMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*w+p - v_coef*d_vals[POS_WMOM];
		flux[POS_RHOT*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*t   - v_coef*d_vals[POS_RHOT];
	}
}

__global__ void do_dir_z_add(double *state, double *tend, double *flux, int n){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;

  if(id == 0){
    //printf("kernel_z_add\n");
  }

	if(id < n){
		int ll= id /(nnz * nnx);
		int k = (id / nnx) % nnz;
		int i = id % nnx;
		int indt  = ll* nnz   * nnx    + k* nnx    + i  ;
		int indf1 = ll*(nnz+1)*(nnx+1) + (k  )*(nnx+1) + i;
		int indf2 = ll*(nnz+1)*(nnx+1) + (k+1)*(nnx+1) + i;
		tend[indt] = -( flux[indf2] - flux[indf1] ) / dz;

    if (ll == POS_WMOM) {
			int inds = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
			tend[indt] = tend[indt] - state[inds]*grav;
    }
	}
}

__global__ void exchange_border_x_1(double *state, int n){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	if(id < n){
		int ll = id / nnz;
		int k = id % nnz;
		int pos = ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs);
		state[pos + 0      ] = state[pos + nnx+hs-2];
		state[pos + 1      ] = state[pos + nnx+hs-1];
		state[pos + nnx+hs  ] = state[pos + hs     ];
		state[pos + nnx+hs+1] = state[pos + hs+1   ];
	}
}

__global__ void exchange_border_x_2(double *state, double *cfd_dens_cell, double *cfd_dens_theta_cell, int n){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	if(id < n){
		int k = id / hs;
		int i = id % hs;
		double z = (k_beg + k+0.5)*dz;
		if (fabs(z-3*zlen/4) <= zlen/16) {
			int ind_r = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
			int ind_u = POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
			int ind_t = POS_RHOT*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
			state[ind_u] = (state[ind_r]+cfd_dens_cell[k+hs]) * 50.;
			state[ind_t] = (state[ind_r]+cfd_dens_cell[k+hs]) * 298. - cfd_dens_theta_cell[k+hs];
		}
	}
}

__global__ void exchange_border_z_1(double *state, int n, int mnt_width){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	if(id < n){
		int ll = id / (nnx+2*hs);
		int i = id % (nnx+2*hs);
		
		int pos = ll*(nnz+2*hs)*(nnx+2*hs);
      
		if (ll == POS_WMOM) {
			state[pos + (0      )*(nnx+2*hs) + i] = 0.;
			state[pos + (1      )*(nnx+2*hs) + i] = 0.;
			state[pos + (nnz+hs  )*(nnx+2*hs) + i] = 0.;
			state[pos + (nnz+hs+1)*(nnx+2*hs) + i] = 0.;
			//Impose the vertical momentum effects of an artificial cos^2 mountain at the lower boundary
			if (config_spec == CONFIG_IN_TEST3) {
			double x = (i_beg+i-hs+0.5)*dx;
			if ( fabs(x-xlen/4) < mnt_width ) {
				double xloc = (x-(xlen/4)) / mnt_width;
				//Compute the derivative of the fake mountain
				double mnt_deriv = -pi*cos(pi*xloc/2)*sin(pi*xloc/2)*10/dx;
				//w = (dz/dx)*u
				state[POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + (0)*(nnx+2*hs) + i] = mnt_deriv*state[POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + hs*(nnx+2*hs) + i];
				state[POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + (1)*(nnx+2*hs) + i] = mnt_deriv*state[POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + hs*(nnx+2*hs) + i];
			}
			}
		} else {
			state[pos + (0      )*(nnx+2*hs) + i] = state[pos + (hs     )*(nnx+2*hs) + i];
			state[pos + (1      )*(nnx+2*hs) + i] = state[pos + (hs     )*(nnx+2*hs) + i];
			state[pos + (nnz+hs  )*(nnx+2*hs) + i] = state[pos + (nnz+hs-1)*(nnx+2*hs) + i];
			state[pos + (nnz+hs+1)*(nnx+2*hs) + i] = state[pos + (nnz+hs-1)*(nnx+2*hs) + i];
		}
	}
}

//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void do_timestep( double *state , double *state_tmp , double *flux , double *tend , double dt ) {

  if (direction_switch) {
    //x-direction first
    do_semi_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    do_semi_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    do_semi_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
    //z-direction second
    do_semi_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    do_semi_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    do_semi_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
  } else {
    //z-direction second
    do_semi_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    do_semi_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    do_semi_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
    //x-direction first
    do_semi_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    do_semi_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    do_semi_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
  }
  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}


//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void do_semi_step( double *state_init , double *state_forcing , double *state_out , double dt , int dir , double *flux , double *tend ) {

  if(dir == DIR_X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    exchange_border_x(state_forcing);
    //Compute the time tendencies for the fluid state in the x-direction
    do_dir_x(state_forcing,flux,tend);
  } else if (dir == DIR_Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    exchange_border_z(state_forcing);
    //Compute the time tendencies for the fluid state in the z-direction
    do_dir_z(state_forcing,flux,tend);
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Apply the tendencies to the fluid state
//   #pragma omp parallel for
//   for (int a = 0; a < NUM_VARS * nnz * nnx; a++) {
//     int ll= a /(nnz * nnx);
//     int k = (a / nnx) % nnz;
//     int i = a % nnx;
//     int inds = (k+hs)*(nnx+2*hs) + ll*(nnz+2*hs)*(nnx+2*hs) + i+hs;
//     int indt = ll*nnz*nnx + k*nnx + i;
//     state_out[inds] = state_init[inds] + dt * tend[indt];
//   }

	int n = NUM_VARS * nnz * nnx;
	int gridSize = (n + blockSize - 1) / blockSize;

	do_semi_step_add<<<gridSize, blockSize>>>(state_out, state_init, tend, n, dt);
  cudaDeviceSynchronize();
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including viscosity)
//Then, compute the tendencies using those fluxes
void do_dir_x( double *state , double *flux , double *tend ) {

  //Compute the hyperviscosity coeficient
  const double v_coef = -hv * dx / (16*dt);;
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
//   #pragma omp parallel for
//   for(int a = 0; a < (nnz) * (nnx + 1); a++){

//     int k = a / (nnx + 1);
//     int i = a % (nnx + 1);
//     //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    
//     double vals[NUM_VARS], d_vals[NUM_VARS];
    
//     for (int ll=0; ll<NUM_VARS; ll++) {
      
//       double stencil[4];
      
//       for (int s=0; s < cfd_size; s++) {
//         int inds = ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+s;
//         stencil[s] = state[inds];
//       }

//       //Fourth-order-accurate interpolation of the state
//       vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
//       //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
//       d_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
//     }

//     //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
//     double r = vals[POS_DENS] + cfd_dens_cell[k+hs];
//     double u = vals[POS_UMOM] / r;
//     double w = vals[POS_WMOM] / r;
//     double t = ( cfd_dens_theta_cell[k+hs] + vals[POS_RHOT] ) / r;
//     double p = pow((r*t),gamm)*C0;

//     //Compute the flux vector
//     flux[POS_DENS*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u     - v_coef*d_vals[POS_DENS];
//     flux[POS_UMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*u+p - v_coef*d_vals[POS_UMOM];
//     flux[POS_WMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*w   - v_coef*d_vals[POS_WMOM];
//     flux[POS_RHOT*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*t   - v_coef*d_vals[POS_RHOT];
//   }

	int n = (nnz) * (nnx + 1);
	int gridSize = (n + blockSize - 1) / blockSize;

	do_dir_x_flux<<<gridSize, blockSize>>>(state, flux, tend, cfd_dens_cell_gpu, cfd_dens_theta_cell_gpu, n, v_coef);
  cudaDeviceSynchronize();

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
//   #pragma omp parallel for
//   for (int a = 0; a < NUM_VARS * nnz * nnx; a++) {
//     int ll= a /(nnz * nnx);
//     int k = (a / nnx) % nnz;
//     int i = a % nnx;
//     int indt  = ll* nnz   * nnx    + k* nnx    + i  ;
//     int indf1 = ll*(nnz+1)*(nnx+1) + k*(nnx+1) + i  ;
//     int indf2 = ll*(nnz+1)*(nnx+1) + k*(nnx+1) + i+1;
//     tend[indt] = -( flux[indf2] - flux[indf1] ) / dx;
//   }

	n = NUM_VARS * nnz * nnx;
	gridSize = (n + blockSize - 1) / blockSize;

	do_dir_x_add<<<gridSize, blockSize>>>(tend, flux, n);
  cudaDeviceSynchronize();
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including viscosity)
//Then, compute the tendencies using those fluxes
void do_dir_z( double *state , double *flux , double *tend ) {

  //Compute the viscosity coeficient
  const double v_coef = -hv * dz / (16 * dt);
  
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
//   #pragma omp parallel for
//   for(int a = 0; a < (nnz + 1) * nnx; a++){

//     int k = a / nnx;
//     int i = a % nnx;
//     //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    
//     double stencil[4], d_vals[NUM_VARS], vals[NUM_VARS];
    
//     for (int ll=0; ll<NUM_VARS; ll++) {
//       for (int s=0; s<cfd_size; s++) {
//         int inds = ll*(nnz+2*hs)*(nnx+2*hs) + (k+s)*(nnx+2*hs) + i+hs;
//         stencil[s] = state[inds];
//       }
//       //Fourth-order-accurate interpolation of the state
//       vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
//       //First-order-accurate interpolation of the third spatial derivative of the state
//       d_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
//     }

//     //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
//     double r = vals[POS_DENS] + cfd_dens_int[k];
//     double u = vals[POS_UMOM] / r;
//     double w = vals[POS_WMOM] / r;
//     double t = ( vals[POS_RHOT] + cfd_dens_theta_int[k] ) / r;
//     double p = C0*pow((r*t),gamm) - cfd_pressure_int[k];
//     //Enforce vertical boundary condition and exact mass conservation
//     if (k == 0 || k == nnz) {
//       w                = 0;
//       d_vals[POS_DENS] = 0;
//     }

//     //Compute the flux vector with viscosity
//     flux[POS_DENS*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w     - v_coef*d_vals[POS_DENS];
//     flux[POS_UMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*u   - v_coef*d_vals[POS_UMOM];
//     flux[POS_WMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*w+p - v_coef*d_vals[POS_WMOM];
//     flux[POS_RHOT*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*t   - v_coef*d_vals[POS_RHOT];
//   }

  int n = (nnz + 1) * nnx;
  int gridSize = (n + blockSize - 1) / blockSize;;

  do_dir_z_flux<<<gridSize, blockSize>>>(state, flux, tend, cfd_dens_int_gpu, cfd_dens_theta_int_gpu, cfd_pressure_int_gpu, n, v_coef);
  cudaDeviceSynchronize();

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
//   #pragma omp parallel for
//   for (int a = 0; a < NUM_VARS * nnz * nnx; a++) {
//     int ll= a /(nnz * nnx);
//     int k = (a / nnx) % nnz;
//     int i = a % nnx;
//     int indt  = ll* nnz   * nnx    + k* nnx    + i  ;
//     int indf1 = ll*(nnz+1)*(nnx+1) + (k  )*(nnx+1) + i;
//     int indf2 = ll*(nnz+1)*(nnx+1) + (k+1)*(nnx+1) + i;
//     tend[indt] = -( flux[indf2] - flux[indf1] ) / dz;
//     if (ll == POS_WMOM) {
//       int inds = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
//       tend[indt] = tend[indt] - state[inds]*grav;
//     }
//   }

  n = NUM_VARS * nnz * nnx;
  gridSize = (n + blockSize - 1) / blockSize;;

  do_dir_z_add<<<gridSize, blockSize>>>(state, tend, flux, n);
  cudaDeviceSynchronize();
}

// CUDA kernel. 
__global__ void copyStatesX(double *d, int n, int nnx_, int nnz_)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n){
        int ll = id / nnz_;
        int k = id % nnz_;

        int pos = ll*(nnz_+4)*(nnx_+4) + (k+hs)*(nnx_+4);

        d[pos      ] = d[pos + nnx_];
        d[pos + 1      ] = d[pos + nnx_+1];
        d[pos + nnx_+2  ] = d[pos + 2     ];
        d[pos + nnx_+3] = d[pos + 3   ];
    }    
}

//Set this MPI task's halo values in the x-direction. This routine will require MPI
void exchange_border_x( double *state ) {
  
  ////////////////////////////////////////////////////////////////////////
  // TODO: EXCHANGE HALO VALUES WITH NEIGHBORING MPI TASKS
  // (1) give    state(1:hs,1:nnz,1:NUM_VARS)       to   my left  neighbor
  // (2) receive state(1-hs:0,1:nnz,1:NUM_VARS)     from my left  neighbor
  // (3) give    state(nnx-hs+1:nnx,1:nnz,1:NUM_VARS) to   my right neighbor
  // (4) receive state(nnx+1:nnx+hs,1:nnz,1:NUM_VARS) from my right neighbor
  ////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////
  // DELETE THE SERIAL CODE BELOW AND REPLACE WITH MPI
  //////////////////////////////////////////////////////
//   for (ll=0; ll<NUM_VARS; ll++) {
//     #pragma omp parallel for
//     for (k=0; k<nnz; k++) {
//       int pos = ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs);
//       state[pos + 0      ] = state[pos + nnx+hs-2];
//       state[pos + 1      ] = state[pos + nnx+hs-1];
//       state[pos + nnx+hs  ] = state[pos + hs     ];
//       state[pos + nnx+hs+1] = state[pos + hs+1   ];
//     }
//   }

	int n = NUM_VARS * nnz;
	int gridSize = (n + blockSize - 1)/blockSize;

	exchange_border_x_1<<<gridSize, blockSize>>>(state, n);
  cudaDeviceSynchronize();
  ////////////////////////////////////////////////////
  
  if (config_spec == CONFIG_IN_TEST6) {
    if (myrank == 0) {
    //   #pragma omp parallel for
    //   for (k=0; k<nnz; k++) {
    //     for (i=0; i<hs; i++) {
    //       z = (k_beg + k+0.5)*dz;
    //       if (fabs(z-3*zlen/4) <= zlen/16) {
    //         ind_r = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
    //         ind_u = POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
    //         ind_t = POS_RHOT*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
    //         state[ind_u] = (state[ind_r]+cfd_dens_cell[k+hs]) * 50.;
    //         state[ind_t] = (state[ind_r]+cfd_dens_cell[k+hs]) * 298. - cfd_dens_theta_cell[k+hs];
    //       }
    //     }
    //   }
      n = nnz * hs;
      gridSize = (n + blockSize - 1)/blockSize;

      exchange_border_x_2<<<gridSize, blockSize>>>(state, cfd_dens_cell_gpu, cfd_dens_theta_cell_gpu, n);
      cudaDeviceSynchronize();
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void exchange_border_z( double *state ) {

  const double mnt_width = xlen/8;
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
//   #pragma omp parallel for
//   for (int ll=0; ll<NUM_VARS; ll++) {
//     for (int i=0; i<nnx+2*hs; i++) {
      
//       int pos = ll*(nnz+2*hs)*(nnx+2*hs);
      
//       if (ll == POS_WMOM) {
//         state[pos + (0      )*(nnx+2*hs) + i] = 0.;
//         state[pos + (1      )*(nnx+2*hs) + i] = 0.;
//         state[pos + (nnz+hs  )*(nnx+2*hs) + i] = 0.;
//         state[pos + (nnz+hs+1)*(nnx+2*hs) + i] = 0.;
//         //Impose the vertical momentum effects of an artificial cos^2 mountain at the lower boundary
//         if (config_spec == CONFIG_IN_TEST3) {
//           double x = (i_beg+i-hs+0.5)*dx;
//           if ( fabs(x-xlen/4) < mnt_width ) {
//             double xloc = (x-(xlen/4)) / mnt_width;
//             //Compute the derivative of the fake mountain
//             double mnt_deriv = -pi*cos(pi*xloc/2)*sin(pi*xloc/2)*10/dx;
//             //w = (dz/dx)*u
//             state[POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + (0)*(nnx+2*hs) + i] = mnt_deriv*state[POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + hs*(nnx+2*hs) + i];
//             state[POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + (1)*(nnx+2*hs) + i] = mnt_deriv*state[POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + hs*(nnx+2*hs) + i];
//           }
//         }
//       } else {
//         state[pos + (0      )*(nnx+2*hs) + i] = state[pos + (hs     )*(nnx+2*hs) + i];
//         state[pos + (1      )*(nnx+2*hs) + i] = state[pos + (hs     )*(nnx+2*hs) + i];
//         state[pos + (nnz+hs  )*(nnx+2*hs) + i] = state[pos + (nnz+hs-1)*(nnx+2*hs) + i];
//         state[pos + (nnz+hs+1)*(nnx+2*hs) + i] = state[pos + (nnz+hs-1)*(nnx+2*hs) + i];
//       }
//     }
//   }
	int n = NUM_VARS * (nnx+2*hs);
	int gridSize = (n + blockSize - 1)/blockSize;

	exchange_border_z_1<<<gridSize, blockSize>>>(state, n, mnt_width);
  cudaDeviceSynchronize();
}


void initialize( int *argc , char ***argv ) {
  int    i, k, ii, kk, ll, inds;
  double x, z, r, u, w, t, hr, ht;

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  //Allocate the model data
  state_cpu              = (double *) malloc( (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double) );
  state_tmp_cpu          = (double *) malloc( (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double) );
  flux_cpu               = (double *) malloc( (nnx+1)*(nnz+1)*NUM_VARS*sizeof(double) );
  tend_cpu               = (double *) malloc( nnx*nnz*NUM_VARS*sizeof(double) );
  cfd_dens_cell_cpu       = (double *) malloc( (nnz+2*hs)*sizeof(double) );
  cfd_dens_theta_cell_cpu = (double *) malloc( (nnz+2*hs)*sizeof(double) );
  cfd_dens_int_cpu        = (double *) malloc( (nnz+1)*sizeof(double) );
  cfd_dens_theta_int_cpu  = (double *) malloc( (nnz+1)*sizeof(double) );
  cfd_pressure_int_cpu    = (double *) malloc( (nnz+1)*sizeof(double) );

  //Allocate GPU memory
  cudaMalloc(&state_gpu, (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double) );
  cudaMalloc(&state_tmp_gpu, (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double) );
  cudaMalloc(&flux_gpu, (nnx+1)*(nnz+1)*NUM_VARS*sizeof(double) );
  cudaMalloc(&tend_gpu, nnx*nnz*NUM_VARS*sizeof(double) );
  cudaMalloc(&cfd_dens_cell_gpu, (nnz+2*hs)*sizeof(double) );
  cudaMalloc(&cfd_dens_theta_cell_gpu, (nnz+2*hs)*sizeof(double) );
  cudaMalloc(&cfd_dens_int_gpu, (nnz+1)*sizeof(double) );
  cudaMalloc(&cfd_dens_theta_int_gpu, (nnz+1)*sizeof(double) );
  cudaMalloc(&cfd_pressure_int_gpu, (nnz+1)*sizeof(double) );

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = dmin(dx,dz) / max_speed * cfl;
  //Set initial elapsed model time 
  etime = 0.;

  //If I'm the master process in MPI, display some grid information
  if (masterproc) {
    printf( "nx_cfd, nz_cfd: %d %d\n", nx_cfd, nz_cfd);
    printf( "dx,dz: %lf %lf\n",dx,dz);
    printf( "dt: %lf\n",dt);
  }

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (k=0; k<nnz+2*hs; k++) {
    for (i=0; i<nnx+2*hs; i++) {
      //Initialize the state to zero
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
        state_cpu[inds] = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a balance + temperature perturbation
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          switch(config_spec){
            case CONFIG_IN_TEST1: 
              testcase1(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST2: 
              testcase2(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST3: 
              testcase3(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST4: 
              testcase4(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST5: 
              testcase5(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST6: 
              testcase6(x,z,r,u,w,t,hr,ht); 
              break;
          }

          //Store into the fluid state array
          inds = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
          state_cpu[inds] = state_cpu[inds] + r                         * qweights[ii]*qweights[kk];
          inds = POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
          state_cpu[inds] = state_cpu[inds] + (r+hr)*u                  * qweights[ii]*qweights[kk];
          inds = POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
          state_cpu[inds] = state_cpu[inds] + (r+hr)*w                  * qweights[ii]*qweights[kk];
          inds = POS_RHOT*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
          state_cpu[inds] = state_cpu[inds] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
        state_tmp_cpu[inds] = state_cpu[inds];
      }
    }
  }
  //Compute the background state over vertical cell averages
  for (k=0; k<nnz+2*hs; k++) {
    cfd_dens_cell_cpu      [k] = 0.;
    cfd_dens_theta_cell_cpu[k] = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (k_beg + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      if (config_spec == CONFIG_IN_TEST1      ) { testcase1      (0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST2        ) { testcase2        (0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST3       ) { testcase3 (0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST4     ) { testcase4     (0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST5) { testcase5(0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST6      ) { testcase6      (0.,z,r,u,w,t,hr,ht); }
      cfd_dens_cell_cpu      [k] = cfd_dens_cell_cpu      [k] + hr    * qweights[kk];
      cfd_dens_theta_cell_cpu[k] = cfd_dens_theta_cell_cpu[k] + hr*ht * qweights[kk];
    }
  }
  //Compute the background state at vertical cell interfaces
  for (k=0; k<nnz+1; k++) {
    z = (k_beg + k)*dz;
    if (config_spec == CONFIG_IN_TEST1      ) { testcase1      (0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST2        ) { testcase2        (0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST3       ) { testcase3 (0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST4     ) { testcase4     (0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST5) { testcase5(0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST6      ) { testcase6      (0.,z,r,u,w,t,hr,ht); }
    cfd_dens_int_cpu      [k] = hr;
    cfd_dens_theta_int_cpu[k] = hr*ht;
    cfd_pressure_int_cpu  [k] = C0*pow((hr*ht),gamm);
  }
}


//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase6( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
}


//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase5( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.);
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase4( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  // call random_number(u);
  // call random_number(w);
  // u = (u-0.5)*20;
  // w = (w-0.5)*20;
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase3( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_bvfreq(z,0.02,hr,ht);
  r = 0.;
  t = 0.;
  u = 15.;
  w = 0.;
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase2( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.);
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase1( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t = t + sample_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}


//Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background density and potential temperature
void const_theta( double z , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p,exner,rt;
  //Establish balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}


//Establish hydrstatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background density and potential temperature
void const_bvfreq( double z , double bv_freq0 , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
double sample_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.) {
    return amp * pow(cos(dist),2.);
  } else {
    return 0.;
  }
}

void finalize() {

  free( state_cpu );
  free( state_tmp_cpu );
  free( flux_cpu );
  free( tend_cpu );
  free( cfd_dens_cell_cpu );
  free( cfd_dens_theta_cell_cpu );
  free( cfd_dens_int_cpu );
  free( cfd_dens_theta_int_cpu );
  free( cfd_pressure_int_cpu );

  cudaFree( state_gpu );
  cudaFree( state_tmp_gpu );
  cudaFree( flux_gpu );
  cudaFree( tend_gpu );
  cudaFree( cfd_dens_cell_gpu );
  cudaFree( cfd_dens_theta_cell_gpu );
  cudaFree( cfd_dens_int_gpu );
  cudaFree( cfd_dens_theta_int_gpu );
  cudaFree( cfd_pressure_int_gpu );
}


//Compute reduced quantities for error checking without resorting
void do_results( double &mass , double &te ) {

  mass = 0;
  te   = 0;
  for (int k=0; k<nnz; k++) {
    for (int i=0; i<nnx; i++) {
      int ind_r = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
      int ind_u = POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
      int ind_w = POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
      int ind_t = POS_RHOT*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
      double r  =   state_cpu[ind_r] + cfd_dens_cell_cpu[hs+k];             // Density
      double u  =   state_cpu[ind_u] / r;                              // U-wind
      double w  =   state_cpu[ind_w] / r;                              // W-wind
      double th = ( state_cpu[ind_t] + cfd_dens_theta_cell_cpu[hs+k] ) / r; // Potential Temperature (theta)
      double p  = C0*pow(r*th,gamm);                               // Pressure
      double t  = th / pow(p0/p,rd/cp);                            // Temperature
      double ke = r*(u*u+w*w);                                     // Kinetic Energy
      double ie = r*cv*t;                                          // Internal Energy
      mass += r        *dx*dz; // Accumulate domain mass
      te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    }
  }
}

void copy_to_gpu(){
  cudaMemcpy(state_gpu, state_cpu, (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(state_tmp_gpu, state_tmp_cpu, (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(flux_gpu, flux_cpu, (nnx+1)*(nnz+1)*NUM_VARS*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(tend_gpu, tend_cpu, nnx*nnz*NUM_VARS*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cfd_dens_cell_gpu, cfd_dens_cell_cpu, (nnz+2*hs)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cfd_dens_theta_cell_gpu, cfd_dens_theta_cell_cpu, (nnz+2*hs)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cfd_dens_int_gpu, cfd_dens_int_cpu, (nnz+1)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cfd_dens_theta_int_gpu, cfd_dens_theta_int_cpu, (nnz+1)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cfd_pressure_int_gpu, cfd_pressure_int_cpu, (nnz+1)*sizeof(double), cudaMemcpyHostToDevice);
}

void copy_to_cpu(){
  cudaMemcpy(state_cpu, state_gpu, (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(state_tmp_cpu, state_tmp_gpu, (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(flux_cpu, flux_gpu, (nnx+1)*(nnz+1)*NUM_VARS*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tend_cpu, tend_gpu, nnx*nnz*NUM_VARS*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cfd_dens_cell_cpu, cfd_dens_cell_gpu, (nnz+2*hs)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cfd_dens_theta_cell_cpu, cfd_dens_theta_cell_gpu, (nnz+2*hs)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cfd_dens_int_cpu, cfd_dens_int_gpu, (nnz+1)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cfd_dens_theta_int_cpu, cfd_dens_theta_int_gpu, (nnz+1)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cfd_pressure_int_cpu, cfd_pressure_int_gpu, (nnz+1)*sizeof(double), cudaMemcpyDeviceToHost);
}

void print(double *v, int n){
  for(int i = 0; i < n; i++)
    if(v[i] != 0.0)
      printf("%d: %lf\n", i, v[i]);
  printf("\n");
}

///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  
  initialize( &argc , &argv );
  cudaDeviceSynchronize();

  //Initial reductions for mass, kinetic energy, and total energy
  do_results(mass0,te0);

  //Copying data to GPU
  copy_to_gpu();
  cudaDeviceSynchronize();

  ////////////////////////////////////////////////////
  // MAIN TIME STEP LOOP
  ////////////////////////////////////////////////////
  auto c_start = std::clock();
  while (etime < sim_time) {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt > sim_time) { dt = sim_time - etime; }
    //Perform a single time step
    do_timestep(state_gpu,state_tmp_gpu,flux_gpu,tend_gpu,dt);
    cudaDeviceSynchronize();
    //Update the elapsed time and output counter
    etime = etime + dt;
    output_counter = output_counter + dt;
    //If it's time for output, reset the counter, and do output
    if (output_counter >= output_freq) {
      output_counter = output_counter - output_freq;
      //Inform the user
      if (masterproc) { 
        printf( "Elapsed Time: %lf / %lf\n", etime , sim_time ); 
      }  
    }
  }
  auto c_end = std::clock();

  if (masterproc) {
    std::cout << "CPU Time: " << ( (double) (c_end-c_start) ) / CLOCKS_PER_SEC << " sec\n";
  }

  copy_to_cpu();
  cudaDeviceSynchronize();

  //Final reductions for mass, kinetic energy, and total energy
  do_results(mass,te);

  if (masterproc) {
    printf( "d_mass: %le\n" , (mass - mass0)/mass0);
    printf( "d_te:   %le\n" , (te   - te0  )/te0   );
  }

  finalize();
}
