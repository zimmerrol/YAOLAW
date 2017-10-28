#pragma once

#include <cblas.h>
#include <complex>

namespace BlasWrapper
{
	inline static void copy(int n, const float* x, int incx, float *y, int incy)
	{
		cblas_scopy(n, x, incx, y, incy);
	}

	inline static void copy(int n, const double* x, int incx, double *y, int incy)
	{
		cblas_dcopy(n, x, incx, y, incy);
	}

	inline static void copy(int n, const std::complex<float>* x, int incx, std::complex<float> *y, int incy)
	{
		cblas_ccopy(n, reinterpret_cast<const float*>(x), incx, reinterpret_cast<float*>(y), incy);
	}

	inline static void copy(int n, const std::complex<double>* x, int incx, std::complex<double> *y, int incy)
	{
		cblas_zcopy(n, reinterpret_cast<const double*>(x), incx, reinterpret_cast<double*>(y), incy);
	}

	inline static void copy(int n, const int *x, int incx, int *y, int incy)
	{
		for (int i = 0; i<n; i++)
		{
			y[i*incy] = x[i*incx];
		}
	}

	inline static void scale(int n, float alpha, float *x, int incx)
	{
		cblas_sscal(n, alpha, x, incx);
	}

	inline static void scale(int n, double alpha, double *x, int incx)
	{
		cblas_dscal(n, alpha, x, incx);
	}

	//! @param trans : gibt an ob A transponiert ist oder nicht. Sei trans = 'N' oder 'n' so ist op(A)= A, sei trans = 'T', 't','C' oder 'c' so ist op(A)= trans(A)
	//! @param m : Anzahl Zeilen in Matrix A.
	//! @param n : Anzahl Spalten in Matrix A.
	//! @param alpha: Skalar fuer A.
	//! @param A : Matrix A
	//! @param lda : leading dimension von A.
	//! @param x : Vektor mit der laenge von mindestens (1+(n-1)*abs(incx)) falls trans = 'N' oder 'n', sonst mindestens der laenge (1+(m-1)*abs(incx)).
	//! @param incx : Speicher Abstand zwischen Elemente in Vector x.
	//! @param beta : Skalar fuer Vektor y.
	//! @param y : Vektor mit der laenge von mindestens (1+(n-1)*abs(incy)) falls trans = 'N' oder 'n', sonst mindestens der laenge (1+(m-1)*abs(incy)).
	//! @param incy: Speicher Abstand zwischen Elemente in Vector y.

	inline static void gemv(char trans, int m, int n, float alpha,
		const float * const A, int lda,
		const float * const x, int incx, float beta,
		float *y, int incy)
	{
		CBLAS_TRANSPOSE tr = (((trans == 't') || (trans == 'T')) ? CblasTrans : CblasNoTrans);
		cblas_sgemv(CblasColMajor, tr, m, n, alpha, A, lda, x, incx, beta, y, incy);
	}

	inline static void gemv(char trans, int m, int n, double alpha,
		const double * const A, int lda,
		const double * const x, int incx, double beta,
		double *y, int incy)
	{
		CBLAS_TRANSPOSE tr = (((trans == 't') || (trans == 'T')) ? CblasTrans : CblasNoTrans);
		cblas_dgemv(CblasColMajor, tr, m, n, alpha, A, lda, x, incx, beta, y, incy);
	}

	inline static void gemv(char trans, int m, int n, std::complex<float> & alpha,
		const std::complex<float> * const A, int lda,
		const std::complex<float> * const x, int incx, std::complex<float> & beta,
		std::complex<float> *y, int incy)
	{
		CBLAS_TRANSPOSE tr = (((trans == 't') || (trans == 'T')) ? CblasTrans : CblasNoTrans);
		//cblas_cgemv(CblasColMajor, tr, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
	}

	inline static void gemv(char trans, int m, int n, std::complex<double> & alpha,
		const std::complex<double> * const A, int lda,
		const std::complex<double> * const x, int incx, std::complex<double>  & beta,
		std::complex<double> *y, int incy)
	{
		CBLAS_TRANSPOSE tr = (((trans == 't') || (trans == 'T')) ? CblasTrans : CblasNoTrans);
		//cblas_zgemv(CblasColMajor, tr, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
	}


	//! @param transa : gibt an ob A transponiert ist oder nicht. Sei transa = 'N' oder 'n' so ist op(A)= A, sei transa = 'T' oder 't' so ist op(A)= trans(A), sei transa = 'C' oder 'c' so ist op(A)=adjoint(A)
	//! @param transb : gibt an ob B transponiert ist oder nicht. Sei transb = 'N' oder 'n' so ist op(B)= A, sei transb = 'T' oder 't' so ist op(B)= trans(B), sei transb = 'C' oder 'c' so ist op(B)=adjoint(B)
	//! @param m : Anzahl Zeilen in Matrix A und Matrix C.
	//! @param n : Anzahl Spalten in Matrix B und Matrix C.
	//! @param k : Anzahl Spalten in Matrix A und Zeilen in Matrix B.
	//! @param alpha: Skalar fuer op(A)*op(B).
	//! @param A : Matrix A
	//! @param lda : leading dimension von A.
	//! @param B : Matrix B.
	//! @param ldb : leading dimension von B.
	//! @param beta : Skalar fuer Matrix C.
	//! @param C : Matrix C.
	//! @param ldc : leading dimension von C.
	inline static void gemm(char transa, char transb, int m, int n, int k, float alpha,
		const float * const A, int lda, const float * const B, int ldb,
		float beta, float * C, int ldc)
	{
		CBLAS_TRANSPOSE tr_a = (((transa == 't') || (transa == 'T')) ? CblasTrans : ((transa == 'c') || (transa == 'C')) ? CblasTrans : CblasNoTrans);
		CBLAS_TRANSPOSE tr_b = (((transb == 't') || (transb == 'T')) ? CblasTrans : ((transb == 'c') || (transb == 'C')) ? CblasTrans : CblasNoTrans);


		cblas_sgemm(CblasColMajor,
			tr_a, tr_b,
			m, n, k,
			alpha,
			A, lda,
			B, ldb,
			beta,
			C, ldc);
	}


	inline static void gemm(char transa, char transb, int m, int n, int k, double alpha,
		const double * const A, int lda, const double * const B, int ldb,
		double beta, double * C, int ldc)
	{
		CBLAS_TRANSPOSE tr_a = (((transa == 't') || (transa == 'T')) ? CblasTrans : ((transa == 'c') || (transa == 'C')) ? CblasTrans : CblasNoTrans);
		CBLAS_TRANSPOSE tr_b = (((transb == 't') || (transb == 'T')) ? CblasTrans : ((transb == 'c') || (transb == 'C')) ? CblasTrans : CblasNoTrans);

		cblas_dgemm(CblasColMajor, tr_a, tr_b, m, n, k, alpha,
			A, lda, B, ldb,
			beta, C, ldc);
	}

	inline static void gemm(char transa, char transb, int m, int n, int k, const std::complex<float> alpha,
		const std::complex<float> * const A, int lda, const std::complex<float> * const B, int ldb,
		const std::complex<float> beta, std::complex<float> * C, int ldc)
	{
		CBLAS_TRANSPOSE tr_a = (((transa == 't') || (transa == 'T')) ? CblasTrans : ((transa == 'c') || (transa == 'C')) ? CblasConjTrans : CblasNoTrans);
		CBLAS_TRANSPOSE tr_b = (((transb == 't') || (transb == 'T')) ? CblasTrans : ((transb == 'c') || (transb == 'C')) ? CblasConjTrans : CblasNoTrans);

		cblas_cgemm(CblasColMajor,
			tr_a, tr_b,
			m, n, k,
			reinterpret_cast<const float*>(&alpha),
			reinterpret_cast<const float*>(A), lda,
			reinterpret_cast<const float*>(B), ldb,
			reinterpret_cast<const float*>(&beta),
			reinterpret_cast<float*>(C), ldc);
	}


	inline static void gemm(char transa, char transb, int m, int n, int k, const std::complex<double> alpha,
		const std::complex<double> * const A, int lda, const std::complex<double> * const B, int ldb,
		const std::complex<double> beta, std::complex<double> * C, int ldc)
	{
		CBLAS_TRANSPOSE tr_a = (((transa == 't') || (transa == 'T')) ? CblasTrans : ((transa == 'c') || (transa == 'C')) ? CblasConjTrans : CblasNoTrans);
		CBLAS_TRANSPOSE tr_b = (((transb == 't') || (transb == 'T')) ? CblasTrans : ((transb == 'c') || (transb == 'C')) ? CblasConjTrans : CblasNoTrans);

		cblas_zgemm(CblasColMajor,
			tr_a, tr_b,
			m, n, k,
			reinterpret_cast<const double*>(&alpha),
			reinterpret_cast<const double*>(A), lda,
			reinterpret_cast<const double*>(B), ldb,
			reinterpret_cast<const double*>(&beta),
			reinterpret_cast<double*>(C), ldc);
	}

	inline static void inv()
	{
		
	}
}