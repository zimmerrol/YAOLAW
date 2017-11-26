#pragma once
#include "la_baseobject.h"
#include <vector>
#include <complex>
#include "vector.h"

namespace LA
{

	template <typename T>
	class Matrix : public LABaseObject<T>
	{
	public:
		Matrix<T>(const size_t nRows, const size_t nCols) : LABaseObject<T>(nRows, nCols) {}
		Matrix<T>() : LABaseObject<T>() {}
		Matrix<T>(const Matrix<T>& _other) : LABaseObject<T>(_other) {}

		Matrix<T>& inv();

		void EVD(std::vector<std::complex<T>>& eigenvalue, Matrix<T>& eigenvectors) const;
		void SVD(Matrix<T>& U, Vector<T>& Sigma, Matrix<T>& Vt) const;
		Matrix<T> transpose();
		Matrix<T> directProduct(const Matrix<T>& rarg) const;

		friend Matrix<T> operator *(const Matrix<T>& larg, const Matrix<T>& rarg)
		{
			Matrix<T> dest;
			dest.resize(larg.getNRows(), rarg.getNCols());

			BlasWrapper::gemm('n', 'n', larg.getNRows(), rarg.getNCols(), larg.getNCols(), 1.0, larg.getDataPtr(), larg.getLeadingDim(), rarg.getDataPtr(),
				rarg.getLeadingDim(), 0.0, dest.getDataPtr(), dest.getLeadingDim());
			return dest;
		}

		friend Matrix<T> operator*(const Matrix<T>& larg, const T& rarg)
		{
			Matrix<T> dest(larg);
			(LABaseObject<T>&)(dest) *= rarg;
			return dest;
		}

		friend Matrix<T> operator+(const Matrix<T>& larg, const Matrix<T>&  rarg)
		{
			Matrix<T> dest(larg);
			(LABaseObject<T>&)(dest) += (const LABaseObject<T>&)rarg;
			return dest;
		}

		static Matrix<T> identity(size_t n);
	};

	template <typename baseT>
	class Matrix<std::complex<baseT>> : public LABaseObject<std::complex<baseT>>
	{
		using T = std::complex<baseT>;

	public:
		Matrix<T>(const size_t nRows, const size_t nCols) : LABaseObject<T>(nRows, nCols) {}
		Matrix<T>() : LABaseObject<T>() {}
		Matrix<T>(const Matrix<T>& _other) : LABaseObject<T>(_other) {}

		Matrix<T>& inv();
		void EVD(std::vector<T>& eigenvalues, Matrix<T>& eigenvectors) const;
		void SVD(Matrix<T>& U, Vector<baseT>& Sigma, Matrix<T>& Vt) const;
		Matrix<T> transpose();
		Matrix<T> adjungate();

		Matrix<T> directProduct(const Matrix<T>& rarg) const;
		static Matrix<T> identity(size_t n);

		friend Matrix<T> operator *(const Matrix<T>& larg, const Matrix<T>& rarg)
		{
			Matrix<T> dest;
			dest.resize(larg.getNRows(), rarg.getNCols());

			BlasWrapper::gemm('n', 'n', larg.getNRows(), rarg.getNCols(), larg.getNCols(), 1.0, larg.getDataPtr(), larg.getLeadingDim(), rarg.getDataPtr(),
				rarg.getLeadingDim(), 0.0, dest.getDataPtr(), dest.getLeadingDim());
			return dest;
		}

		friend Matrix<T> operator*(const Matrix<T>& larg, const T& rarg)
		{
			Matrix<T> dest(larg);
			(LABaseObject<T>&)(dest) *= rarg;
			return dest;
		}

		friend Matrix<T> operator+(const Matrix<T>& larg, const Matrix<T>&  rarg)
		{
			Matrix<T> dest(larg);
			(LABaseObject<T>&)(dest) += (const LABaseObject<T>&)rarg;
			return dest;
		}
	};

}

