#pragma once

#include <complex>

#include <cblas.h>



#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

#include <lapacke.h>


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
		for (int i = 0; i < n; i++)
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

	inline static void scale(int n, const std::complex<float> alpha, std::complex<float> *x, int incx)
	{
		cblas_cscal(n, reinterpret_cast<const float*>(&alpha), reinterpret_cast<float*>(x), incx);
	}

	inline static void scale(int n, const std::complex<double> alpha, std::complex<double> *x, int incx)
	{
		cblas_zscal(n, reinterpret_cast<const double*>(&alpha), reinterpret_cast<double*>(x), incx);
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

	inline static void omatcopy(char trans, int crows, int ccols, float calpha, const float* a, int clda, float* b, int cldb)
	{
		CBLAS_TRANSPOSE tr = (((trans == 't') || (trans == 'T')) ? CblasTrans : ((trans == 'c') || (trans == 'C')) ? CblasConjTrans : CblasNoTrans);
		cblas_somatcopy(CBLAS_ORDER::CblasColMajor, tr, crows, ccols, calpha, a, clda, b, cldb);
	}

	inline static void omatcopy(char trans, int crows, int ccols, double calpha, const double* a, int clda, double* b, int cldb)
	{
		CBLAS_TRANSPOSE tr = (((trans == 't') || (trans == 'T')) ? CblasTrans : ((trans == 'c') || (trans == 'C')) ? CblasConjTrans : CblasNoTrans);
		cblas_domatcopy(CBLAS_ORDER::CblasColMajor, tr, crows, ccols, calpha, a, clda, b, cldb);
	}

	inline static void omatcopy(char trans, int crows, int ccols, const std::complex<float> calpha, const std::complex<float>* a, int clda, std::complex<float>* b, int cldb)
	{
		CBLAS_TRANSPOSE tr = (((trans == 't') || (trans == 'T')) ? CblasTrans : ((trans == 'c') || (trans == 'C')) ? CblasConjTrans : CblasNoTrans);
		cblas_comatcopy(CBLAS_ORDER::CblasColMajor, tr, crows, ccols, reinterpret_cast<const float*>(&calpha),
			reinterpret_cast<const float*>(&a), clda, reinterpret_cast<float*>(&b), cldb);
	}

	inline static void omatcopy(char trans, int crows, int ccols, const std::complex<double> calpha, const std::complex<double>* a, int clda, std::complex<double>* b, int cldb)
	{
		CBLAS_TRANSPOSE tr = (((trans == 't') || (trans == 'T')) ? CblasTrans : ((trans == 'c') || (trans == 'C')) ? CblasConjTrans : CblasNoTrans);
		cblas_zomatcopy(CBLAS_ORDER::CblasColMajor, tr, crows, ccols, reinterpret_cast<const double*>(&calpha),
			reinterpret_cast<const double*>(&a), clda, reinterpret_cast<double*>(&b), cldb);
	}

	inline static int geev(char jobvl, char jobvr, int N, float* A, int lda, float* wr, float* wi, float* vl, int ldvl, float* vr, int ldvr)
	{
		return LAPACKE_sgeev(CblasColMajor, jobvl, jobvr, N, A, lda, wr, wi, vl, ldvl, vr, ldvr);
	}

	inline static int geev(char jobvl, char jobvr, int N, double* A, int lda, double* wr, double* wi, double* vl, int ldvl, double* vr, int ldvr)
	{
		return LAPACKE_dgeev(CblasColMajor, jobvl, jobvr, N, A, lda, wr, wi, vl, ldvl, vr, ldvr);
	}

	inline static int geev(char jobvl, char jobvr, int N, std::complex<float>* A, int lda, std::complex<float>* w, std::complex<float>* vl, int ldvl, std::complex<float>* vr, int ldvr)
	{
		return LAPACKE_cgeev(CblasColMajor, jobvl, jobvr, N, A, lda, w, vl, ldvl, vr, ldvr);
	}

	inline static int geev(char jobvl, char jobvr, int N, std::complex<double>* A, int lda, std::complex<double>* w, std::complex<double>* vl, int ldvl, std::complex<double>* vr, int ldvr)
	{
		return LAPACKE_zgeev(CblasColMajor, jobvl, jobvr, N, A, lda, w, vl, ldvl, vr, ldvr);
	}


	inline static int gesvd(char jobu, char jobvt, int m, int n, float* A, int lda, float* s, float* u, int ldu, float* vt, int ldvt, float* superb)
	{
		return LAPACKE_sgesvd(CblasColMajor, jobu, jobvt, m, n, A, lda, s, u, ldu, vt, ldvt, superb);
	}

	inline static int gesvd(char jobu, char jobvt, int m, int n, double* A, int lda, double* s, double* u, int ldu, double* vt, int ldvt, double* superb)
	{
		return LAPACKE_dgesvd(CblasColMajor, jobu, jobvt, m, n, A, lda, s, u, ldu, vt, ldvt, superb);
	}

	inline static int gesvd(char jobu, char jobvt, int m, int n, std::complex<float>* A, int lda, float* s, std::complex<float>* u,
		int ldu, std::complex<float>* vt, int ldvt, float* superb)
	{
		return LAPACKE_cgesvd(CblasColMajor, jobu, jobvt, m, n, A, lda, s, u, ldu, vt, ldvt, superb);
	}

	inline static int gesvd(char jobu, char jobvt, int m, int n, std::complex<double>* A, int lda, double* s, std::complex<double>* u,
		int ldu, std::complex<double>* vt, int ldvt, double* superb)
	{
		return LAPACKE_zgesvd(CblasColMajor, jobu, jobvt, m, n, A, lda, s, u, ldu, vt, ldvt, superb);
	}

	inline static float dot(int n, const float* x, int incx, const float* y, int incy)
	{
		return cblas_sdot(n, x, incx, y, incy);
	}

	inline static double dot(int n, const double* x, int incx, const double* y, int incy)
	{
		return cblas_ddot(n, x, incx, y, incy);
	}

	inline static std::complex<float> dot(int n, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy)
	{
		auto result = cblas_cdotu(n, reinterpret_cast<const float*>(x), incx, reinterpret_cast<const float*>(y), incy);
		return std::complex<float>(result.real, result.imag);
	}

	inline static std::complex<double> dot(int n, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy)
	{
		auto result = cblas_zdotu(n, reinterpret_cast<const double*>(x), incx, reinterpret_cast<const double*>(y), incy);
		return std::complex<double>(result.real, result.imag);
	}

}