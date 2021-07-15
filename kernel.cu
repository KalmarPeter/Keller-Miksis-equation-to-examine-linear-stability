﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>

#define Resolution 16384
#define Res 128
#define NumberOfModes 5
#define SysDim 12
# define Perturbation 1.0e-6
#define T 1.0
#define AbsTolerance 1.0e-9
#define RelTolerance 1.0e-9
#define AbsToleranceModes 1.0e-8
#define RelToleranceModes 1.0e-8
#define ConvergedIteration 90
#define TransientIteration 512
#define GrowLimit  5.0
#define ShrinkLimit  0.1

// Functions

void Samplog(double, double, int, double*);
__global__ void GetC(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void GetS(double*, double*);
__forceinline__ __device__ void OdeFun(double*, double&, double*, double*, double*, double*);
__forceinline__ __device__ void rkck(double*, double*, double*, double&, double*, double*, double*);
__forceinline__ __device__ void GetTolerance(double*, double*, double*);
__forceinline__ __device__ void GetTimeStep(double*, double*, double*, double*, double*, double&, double&, double*);
__global__ void OdeSolver(double*, double*, double*, double*, double*, double*);


clock_t SimulationStart = clock();

//Control parameters:

double R0[] = { (10.0e-6) }; // equilibrium bouble size
const int BlockSize = 32;
const int GridSize = Resolution / BlockSize + (Resolution % BlockSize == 0 ? 0 : 1);

// Features of the fluid:

double penv[] = { 1.0e5 }; // environmental pressure
double pv[] = { 3166.8 }; // vapor pressure
double n[] = { 1.4 }; // polytropic exponent
double rho[] = { 997.0 }; // density
double sft[] = { 0.072 }; // surface tension
double nu[] = { 0.00089 }; // viscosity
double cl[] = { 1497.0 }; // speed of sound in water
double theta[] = { 0.0 };

const double fr0 = (1.0 / (2.0 * M_PI * *R0 * pow(*rho, (1.0 / 2.0)))) * pow((3.0 * *n * (*penv + (2.0 * *sft / *R0) - *pv) - (2.0 * *sft / *R0) - (4.0 * pow(*nu, 2.0)) / (*rho * pow(*R0, 2.0))), (1.0 / 2.0)); // own frequency of the bubble
const double omega0 = pow(((3.0 * *n * (*penv - *pv)) / (*rho * *R0 * *R0)) + ((2.0 * (3.0 * *n - 1.0) * *sft) / (*rho * *R0)), (1.0 / 2.0)); // own angular speed of the bubble

int main()
{
	// Initial Conditions

	double* h_t0 = new double[Resolution];
	double* h_y = new double[SysDim * Resolution];
	double* h_Rm = new double[Resolution];
	double* h_l = new double[NumberOfModes];
	double* h_s = new double[5 * NumberOfModes];
	double* h_r = new double[NumberOfModes * Resolution];

	double* d_t0;
	double* d_y;
	double* d_Rm;
	double* d_s;
	double* d_r;

	cudaMalloc((void**)&d_t0, Resolution * sizeof(double));
	cudaMalloc((void**)&d_y, SysDim * Resolution * sizeof(double));
	cudaMalloc((void**)&d_Rm, Resolution * sizeof(double));
	cudaMalloc((void**)&d_s, 5 * NumberOfModes * sizeof(double));
	cudaMalloc((void**)&d_r, NumberOfModes * Resolution * sizeof(double));

	for (int i = 0; i < Resolution; i++)
	{
		h_t0[i] = 0.0;
		h_y[i] = 1.0;
		h_y[i + Resolution] = 0.0;
		h_Rm[i] = *R0;
	}

	for (int i = 0; i < Resolution * NumberOfModes; i++)
	{
		h_r[i] = 0.0;
	}

	for (int x = 0; x < NumberOfModes; x++)
	{
		for (int i = 0; i < Resolution; i++)
		{
			h_y[i + (x + 1) * 2 * Resolution] = Perturbation;
			h_y[i + (2 * x + 3) * Resolution] = 0.0;
		}
	}

	h_l[0] = 2.0;
	h_l[1] = 3.0;
	h_l[2] = 4.0;
	h_l[3] = 5.0;
	h_l[4] = 6.0;

	cudaMemcpy(d_t0, h_t0, Resolution * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, SysDim * Resolution * sizeof(double), cudaMemcpyHostToDevice);

	// Copy fluid properties to device

	double* d_penv;
	double* d_pv;
	double* d_n;
	double* d_rho;
	double* d_sft;
	double* d_nu;
	double* d_cl;
	double* d_theta;
	double* d_R0;

	cudaMalloc((void**)&d_penv, sizeof(double));
	cudaMalloc((void**)&d_pv, sizeof(double));
	cudaMalloc((void**)&d_n, sizeof(double));
	cudaMalloc((void**)&d_rho, sizeof(double));
	cudaMalloc((void**)&d_sft, sizeof(double));
	cudaMalloc((void**)&d_nu, sizeof(double));
	cudaMalloc((void**)&d_cl, sizeof(double));
	cudaMalloc((void**)&d_theta, sizeof(double));
	cudaMalloc((void**)&d_R0, sizeof(double));

	cudaMemcpy(d_penv, penv, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pv, pv, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, n, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, rho, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sft, sft, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nu, nu, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cl, cl, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_theta, theta, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R0, R0, sizeof(double), cudaMemcpyHostToDevice);

	// Parameters of the excitation

	double pa1[] = { 0.6e5 }; // pressure amplitude 1
	double pa2[] = { 0.4e5 }; // pressure amplitude 2
	double* omega1 = new double[Resolution]; // angular speed 1
	double* omega2 = new double[Resolution]; // angular speed 2
	double* fr1 = new double[Res];
	double* fr2 = new double[Res];
	double* Fr1 = new double[Resolution];
	double* Fr2 = new double[Resolution];

	// Logaritmic sampling

	double LowerBoundary = 20.0e3;
	double UpperBoundary = 2.0e6;
	Samplog(LowerBoundary, UpperBoundary, Res, fr1);
	Samplog(LowerBoundary, UpperBoundary, Res, fr2);

	// Copy the exacition parameters to the device

	for (int i = 0; i < Res; i++)
	{
		for (int j = 0; j < Res; j++)
		{
			Fr1[j + i * Res] = fr1[j];
			Fr2[j + i * Res] = fr2[i];
			omega1[j + i * Res] = 2.0 * M_PI * fr1[j];
			omega2[j + i * Res] = 2.0 * M_PI * fr2[i];
		}
	}

	double* d_pa1;
	double* d_pa2;
	double* d_omega1;
	double* d_omega2;

	cudaMalloc((void**)&d_pa1, sizeof(double));
	cudaMalloc((void**)&d_pa2, sizeof(double));
	cudaMalloc((void**)&d_omega1, Resolution * sizeof(double));
	cudaMalloc((void**)&d_omega2, Resolution * sizeof(double));

	cudaMemcpy(d_pa1, pa1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pa2, pa2, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_omega1, omega1, Resolution * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_omega2, omega2, Resolution * sizeof(double), cudaMemcpyHostToDevice);

	// Create a txt

	std::ofstream DataFile;
	DataFile.open("rkck_cuda.txt");
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);

	// Define d_c pointer for GetC function

	double* d_c;
	cudaMalloc((void**)&d_c, 17 * Resolution * sizeof(double));

	// Get constans for OdeFun

	GetC << <GridSize, BlockSize >> > (d_omega1, d_omega2, d_c, d_penv, d_pv, d_R0, d_rho, d_cl, d_n, d_sft, d_pa1, d_pa2, d_nu, d_theta);

	GetS(h_s, h_l);

	// Ode Solver

	cudaMemcpy(d_s, h_s, 5 * NumberOfModes * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	for (int z = 0; z < TransientIteration; z++)
	{
		OdeSolver << <GridSize, BlockSize >> > (d_t0, d_y, d_c, d_s, d_Rm, d_r);
		cudaDeviceSynchronize();
		std::cout << z << " t" << std::endl;
	}


	cudaMemcpy(d_Rm, h_Rm, sizeof(double) * Resolution, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, sizeof(double) * NumberOfModes * Resolution, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	for (int z = 0; z < ConvergedIteration; z++)
	{
		OdeSolver << <GridSize, BlockSize >> > (d_t0, d_y, d_c, d_s, d_Rm, d_r);
		cudaDeviceSynchronize();
		std::cout << z << " c" << std::endl;
	}

	cudaMemcpy(h_r, d_r, Resolution * NumberOfModes * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Rm, d_Rm, Resolution * sizeof(double), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	//Write maximum bubble radius at the given sampleing to txt

	for (int j = 0; j < NumberOfModes; j++)
	{
		for (int i = 0; i < Resolution; ++i)
		{
			DataFile.width(Width); DataFile << (h_r[i + j * Resolution] / ConvergedIteration) * Fr1[i] / fr0 << ',';
			DataFile.width(Width); DataFile << Fr1[i] << ',';
			DataFile.width(Width); DataFile << h_Rm[i] << ',';
			DataFile.width(Width); DataFile << Fr2[i];
			DataFile << '\n';
		}
	}

	DataFile.close();

	// Delete pointers

	cudaFree(d_omega1);
	cudaFree(d_omega2);
	cudaFree(d_c);
	cudaFree(d_penv);
	cudaFree(d_pv);
	cudaFree(d_R0);
	cudaFree(d_rho);
	cudaFree(d_cl);
	cudaFree(d_n);
	cudaFree(d_sft);
	cudaFree(d_pa1);
	cudaFree(d_pa2);
	cudaFree(d_nu);
	cudaFree(d_theta);
	cudaFree(d_t0);
	cudaFree(d_y);
	cudaFree(d_Rm);
	cudaFree(d_s);
	cudaFree(d_r);

	//delete[] omega1, omega2, pa1,pa2,h_t0,h_y;
	//delete[] penv, pv, R0, rho, cl, n, sft, nu, theta;

	clock_t SimulationEnd = clock();
	std::cout << 1000 * (SimulationEnd - SimulationStart) / CLOCKS_PER_SEC << std::endl;

	return 0;
}

// Logaritmic sampling

void Samplog(double LowerBoundary, double UpperBoundary, int NumberOfIntervals, double* Intervals)
{
	double LogUpper = log10(UpperBoundary);
	double LogLower = log10(LowerBoundary);
	double Steps = (LogUpper - LogLower) / (NumberOfIntervals - 1.0);

	Intervals[0] = LowerBoundary;
	Intervals[NumberOfIntervals - 1] = UpperBoundary;

	for (int i = 1; i < NumberOfIntervals - 1; i++)
	{
		Intervals[i] = pow(10, LogLower + i * Steps);
	}
}

// Constans for dimensionless Ryleigh Plesset equation

__global__ void GetC(double* d_omega1, double* d_omega2, double* d_c, double* d_penv, double* d_pv, double* d_R0, double* d_rho, double* d_cl, double* d_n, double* d_sft, double* d_pa1, double* d_pa2, double* d_nu, double* d_theta)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double r_pa1 = *d_pa1;
	double r_pa2 = *d_pa2;
	double r_penv = *d_penv;
	double r_pv = *d_pv;
	double r_R0 = *d_R0;
	double r_rho = *d_rho;
	double r_cl = *d_cl;
	double r_n = *d_n;
	double r_sft = *d_sft;
	double r_nu = *d_nu;
	double r_theta = *d_theta;

	d_c[tid] = (((4.0 * M_PI * M_PI) / (r_R0 * r_R0 * d_omega1[tid] * d_omega1[tid] * r_rho)) * (((2.0 * r_sft) / (r_R0)) + r_penv - r_pv));
	d_c[tid + Resolution] = ((1.0 - 3.0 * r_n) * (2 * r_sft / r_R0 + r_penv - r_pv) * (2.0 * M_PI / r_R0 / d_omega1[tid]) / r_cl / r_rho);
	d_c[tid + 2 * Resolution] = (((r_penv - r_pv) * 4.0 * M_PI * M_PI) / (r_R0 * r_R0 * d_omega1[tid] * d_omega1[tid] * r_rho));
	d_c[tid + 3 * Resolution] = ((8.0 * M_PI * M_PI * r_sft) / (r_R0 * r_R0 * r_R0 * d_omega1[tid] * d_omega1[tid] * r_rho));
	d_c[tid + 4 * Resolution] = ((8.0 * M_PI * r_nu) / (r_R0 * r_R0 * d_omega1[tid] * r_rho));
	d_c[tid + 5 * Resolution] = ((4.0 * M_PI * M_PI * r_pa1) / (r_R0 * r_R0 * d_omega1[tid] * d_omega1[tid] * r_rho));
	d_c[tid + 6 * Resolution] = ((4.0 * M_PI * M_PI * r_pa2) / (r_R0 * r_R0 * d_omega1[tid] * d_omega1[tid] * r_rho));
	d_c[tid + 7 * Resolution] = ((4.0 * M_PI * M_PI * r_pa1) / (r_R0 * d_omega1[tid] * r_rho * r_cl));
	d_c[tid + 8 * Resolution] = ((4.0 * M_PI * M_PI * d_omega2[tid] * r_pa2) / (r_R0 * d_omega1[tid] * d_omega1[tid] * r_rho * r_cl));
	d_c[tid + 9 * Resolution] = ((r_R0 * d_omega1[tid]) / (2.0 * M_PI * r_cl));
	d_c[tid + 10 * Resolution] = (3.0 * r_n);
	d_c[tid + 11 * Resolution] = ((d_omega2[tid]) / (d_omega1[tid]));
	d_c[tid + 12 * Resolution] = (r_theta);
	d_c[tid + 13 * Resolution] = (2.0 * M_PI / d_omega1[tid]);
	d_c[tid + 14 * Resolution] = r_R0;

	if (r_pa1 != 0.0)
	{
		d_c[tid + 15 * Resolution] = (sqrt(r_nu / r_rho / d_omega1[tid]));
	}
	else
	{
		d_c[tid + 15 * Resolution] = 0.0;
	}

	if (r_pa2 != 0.0)
	{
		d_c[tid + 16 * Resolution] = (sqrt(r_nu / r_rho / d_omega2[tid]));
	}
	else
	{
		d_c[tid + 16 * Resolution] = 0.0;
	}

}

void GetS(double* h_s, double* h_l)
{
	for (int i = 0; i < NumberOfModes; i++)
	{
		h_s[5 * i] = h_l[i];
		h_s[5 * i + 1] = h_l[i] - 1.0;
		h_s[5 * i + 2] = h_l[i] * (h_l[i] + 2.0) * (h_l[i] + 2.0);
		h_s[5 * i + 3] = h_l[i] * (h_l[i] - 1.0) * (h_l[i] + 2.0);
		h_s[5 * i + 4] = (h_l[i] - 1.0) * (h_l[i] + 1.0) * (h_l[i] + 2.0);
	}
}

// Dimensionless Keller - Miksis equation

__forceinline__ __device__ void OdeFun(double* f, double& dt, double* t0, double* y, double* c, double* d_s)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double rply;
	double N;
	double D;
	double arg1;
	double arg2;
	double NPerD;
	rply = 1.0 / y[tid];
	arg1 = 2.0 * M_PI * t0[tid];
	arg2 = 2.0 * M_PI * c[tid + 11 * Resolution] * t0[tid] + c[tid + 12 * Resolution];

	f[0] = dt * y[tid + Resolution];

	N = (c[tid] + c[tid + Resolution] * y[tid + Resolution]) * pow(rply, c[tid + 10 * Resolution]) - c[tid + 2 * Resolution] * (1.0 + c[tid + 9 * Resolution] * y[tid + Resolution]) - c[tid + 3 * Resolution] * rply - c[tid + 4 * Resolution] * y[tid + Resolution] * rply
		- (1.5 - 0.5 * c[tid + 9 * Resolution] * y[tid + Resolution]) * y[tid + Resolution] * y[tid + Resolution] - (c[tid + 5 * Resolution] * sin(arg1) + c[tid + 6 * Resolution] * sin(arg2)) * (1.0 + c[tid + 9 * Resolution] * y[tid + Resolution])
		- y[tid] * (c[tid + 7 * Resolution] * cos(arg1) + c[tid + 8 * Resolution] * cos(arg2));

	D = y[tid] - c[tid + 9 * Resolution] * y[tid] * y[tid + Resolution] + c[tid + 4 * Resolution] * c[tid + 9 * Resolution];

	NPerD = N / D;

	f[1] = dt * NPerD;

	// Modes

	int t = threadIdx.x;

	__shared__ double s_s[5 * NumberOfModes];

	if (t < 5 * NumberOfModes)
	{
		s_s[t] = d_s[t];
	}
	__syncthreads();

	double blt[NumberOfModes]; // boundary layer thickness
	double A[NumberOfModes];
	double B[NumberOfModes];
	double E[NumberOfModes];
	double yc14;


	for (int i = 0; i < NumberOfModes; i++)
	{
		yc14 = (y[tid] * c[tid + 14 * Resolution]);
		blt[i] = 2.0 * (fmin((fmax(c[tid + 15 * Resolution], c[tid + 16 * Resolution])), yc14 / (2.0 * s_s[5 * i])));
		E[i] = (1.0 / (1.0 + blt[i] / yc14));
	}

	for (int i = 0; i < NumberOfModes; i++)
	{
		A[i] = (-s_s[5 * i + 1] * NPerD * rply + s_s[5 * i + 4] * c[tid + 3 * Resolution] * 0.5 * rply * rply * rply + 0.5 * c[tid + 4 * Resolution] * (s_s[5 * i + 4] - s_s[5 * i + 3] * E[i]) * y[tid + Resolution] * rply * rply * rply);

		B[i] = (3.0 * y[tid + Resolution] * rply + 0.5 * c[tid + 4 * Resolution] * rply * rply * (s_s[5 * i + 2] * E[i] - s_s[5 * i + 4]));
	}

	for (int i = 0; i < NumberOfModes; i++)
	{
		f[2 * i + 2] = dt * (y[tid + Resolution * (2 * i + 3)]);

		f[2 * i + 3] = dt * (-A[i] * y[tid + 2 * Resolution * (i + 1)] - B[i] * y[tid + Resolution * (2 * i + 3)]);
	}
}

// Runge-Kutta-Cash-Karp Method

__forceinline__ __device__ void rkck(double* t0, double* y, double* c, double& dt, double* yn, double* error, double* d_s)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double k1[SysDim];
	double k2[SysDim];
	double k3[SysDim];
	double k4[SysDim];
	double k5[SysDim];
	double k6[SysDim];
	double yact[SysDim];


	//k1

	OdeFun(k1, dt, t0, y, c, d_s);

	//k2

	double hk2[1];
	hk2[0] = t0[tid] + (1.0 / 5.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[tid + i * Resolution] = y[tid + i * Resolution] + (1.0 / 5.0) * k1[i];
	}

	OdeFun(k2, dt, hk2, yact, c, d_s);

	//k3

	double hk3[1];
	hk3[0] = t0[tid] + (3.0 / 10.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[tid + i * Resolution] = y[tid + i * Resolution] + (3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i];
	}

	OdeFun(k3, dt, hk3, yact, c, d_s);

	//k4

	double hk4[1];
	hk4[0] = t0[tid] + (3.0 / 5.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[tid + i * Resolution] = y[tid + i * Resolution] + (3.0 / 10.0) * k1[i] + (-9.0 / 10.0) * k2[i] + (6.0 / 5.0) * k3[i];
	}

	OdeFun(k4, dt, hk4, yact, c, d_s);

	//k5

	double hk5[1];
	hk5[0] = t0[tid] + (1.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[tid + i * Resolution] = y[tid + i * Resolution] + (-11.0 / 54.0) * k1[i] + (5.0 / 2.0) * k2[i] + (-70.0 / 27.0) * k3[i] + (35.0 / 27.0) * k4[i];
	}

	OdeFun(k5, dt, hk5, yact, c, d_s);

	//k6

	double hk6[1];
	hk6[0] = t0[tid] + (7.0 / 8.0) * dt;

	for (int i = 0; i < SysDim; ++i)
	{
		yact[tid + i * Resolution] = y[tid + i * Resolution] + (1631.0 / 55296.0) * k1[i] + (175.0 / 512.0) * k2[i] + (575.0 / 13824.0) * k3[i] + (44275.0 / 110592.0) * k4[i] + (253.0 / 4096.0) * k5[i];
	}

	OdeFun(k6, dt, hk6, yact, c, d_s);

	// yn

	for (int i = 0; i < SysDim; ++i)
	{
		yn[tid + i * Resolution] = y[tid + i * Resolution] + (37.0 / 378.0) * k1[i] + (0.0) * k2[i] + (250.0 / 621.0) * k3[i] + (125.0 / 594.0) * k4[i] + (0.0) * k5[i] + (512.0 / 1771.0) * k6[i];
	}

	// error

	for (int i = 0; i < SysDim; ++i)
	{
		error[tid + i * Resolution] = fabs(((37.0 / 378.0) - (2825.0 / 27648.0)) * k1[i] + (0.0) * k2[i] + ((250.0 / 621.0) - (18575.0 / 48384.0)) * k3[i] + ((125.0 / 594.0) - (13525.0 / 55296.0)) * k4[i] + ((0.0) - (277.0 / 14336.0)) * k5[i] + ((512.0 / 1771.0) - (1.0 / 4.0)) * k6[i]) + 1.0e-30;
	}
}

//Define tolerances

__forceinline__ __device__ void GetTolerance(double* y, double* yn, double* tol)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double abstol[SysDim];
	double reltol[SysDim];
	double reltolacty[SysDim];
	double reltolactyn[SysDim];
	double reltolact[SysDim];
	double habs = 1.0e-300;
	double hrelact = 1.0e-300;

	for (int i = 0; i < 2; i++)
	{
		abstol[i] = AbsTolerance;
		reltol[i] = RelTolerance;
	}

	for (int i = 2; i < SysDim; i++)
	{
		abstol[i] = AbsToleranceModes;
		reltol[i] = RelToleranceModes;
	}

	for (int x = 0; x < SysDim; ++x)
	{
		reltolacty[x] = reltol[x] * fabs(y[tid + x * Resolution]);
		reltolactyn[x] = reltol[x] * fabs(yn[tid + x * Resolution]);
	}

	for (int x = 0; x < SysDim; x++)
	{
		reltolact[x] = fmin(reltolacty[x], reltolactyn[x]);
	}


	for (int i = 0; i < SysDim; ++i)
	{
		if (abstol[i] > habs)
		{
			habs = abstol[i];
		}

		if (reltolact[i] > hrelact)
		{
			hrelact = reltolact[i];
		}
	}

	if (habs >= hrelact)
	{
		for (int x = 0; x < SysDim; ++x)
		{
			tol[tid + x * Resolution] = abstol[x];
		}
	}

	else
	{
		for (int x = 0; x < SysDim; ++x)
		{
			tol[tid + x * Resolution] = reltolact[x];
		}
	}
}

//Calculate the following time step

__forceinline__ __device__ void GetTimeStep(double* tol, double* error, double* y, double* yn, double* t0, double& dt, double& t, double* Rm)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	double MintolDivError = 1.0e300;
	double TimeStepper;
	double MaxTimeStep = 1.0e6;
	double MinTimeStep = 1.0e-12;
	bool Update = false;

	for (int i = 0; i < SysDim; ++i)
	{
		if ((tol[tid + i * Resolution] / error[tid + i * Resolution]) < MintolDivError)
		{
			MintolDivError = (tol[tid + i * Resolution] / error[tid + i * Resolution]);
		}
	}

	if (MintolDivError >= 1)
	{
		Update = 1;
	}

	if (Update == 1)
	{
		TimeStepper = 0.9 * pow(MintolDivError, 0.2);
	}

	else
	{
		TimeStepper = 0.9 * pow(MintolDivError, 0.25);
	}

	if (isfinite(TimeStepper) == 0)
	{
		Update = 0;
	}

	if (Update == 1)
	{
		for (int x = 0; x < SysDim; ++x)
		{
			y[tid + x * Resolution] = yn[tid + x * Resolution];
		}

		t0[tid] += dt;
		t += dt;
	}

	TimeStepper = fmin(TimeStepper, GrowLimit);
	TimeStepper = fmax(TimeStepper, ShrinkLimit);

	dt = dt * TimeStepper;

	dt = fmin(dt, MaxTimeStep);
	dt = fmax(dt, MinTimeStep);

	if ((t + dt) > T)
	{
		dt = T - t;
	}

	if (y[tid] > Rm[tid])
	{
		Rm[tid] = y[tid];
	}
}

// ODE Solver

__global__ void OdeSolver(double* d_t0, double* d_y, double* c, double* d_s, double* d_Rm, double* d_r)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double r_t = 0.0;
	double r_dt = 3.016e-6;
	double r_yn[SysDim];
	double r_error[SysDim];
	double r_tol[SysDim];
	double r_r[NumberOfModes];

	while (r_t < T)
	{
		rkck(d_t0, d_y, c, r_dt, r_yn, r_error, d_s);
		GetTolerance(d_y, r_yn, r_tol);
		GetTimeStep(r_tol, r_error, d_y, r_yn, d_t0, r_dt, r_t, d_Rm);
	}

	// Define initial perturbation for modes, caculate grow rate

	for (int i = 0; i < NumberOfModes; i++)
	{
		r_r[i] = fabs(d_y[tid + (i + 1) * 2 * Resolution] / Perturbation);

		if (r_r[i] > 0.0)
		{
			d_y[tid + 2 * (i + 1) * Resolution] /= r_r[i];
			d_y[tid + (2 * i + 3) * Resolution] /= r_r[i];

			d_r[tid + i * Resolution] += log(r_r[i]);
		}

	}
}