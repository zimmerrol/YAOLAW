#pragma once
#include "la_baseobject.h"
#include "la_operations.h"

template <typename T>
class Matrix : public LABaseObject<T>
{

public:

	Matrix<T>(const size_t nRows, const size_t nCols) : LABaseObject<T>(nRows, nCols) {}

	Matrix<T>() : LABaseObject<T>() {}

	Matrix<T>(const Matrix<T>& _other) : LABaseObject<T>(_other) {}

	Matrix<T>& operator=(const Matrix<T>& source)
	{
		this->copyDataFrom(source);
		return *this;
	}

	Matrix<T>& operator *= (const T& _value);

	Matrix<T>& inv();

	template <typename T>
	friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& source)
	{
		source.print(os);
		return os;
	}

	template <typename T>
	friend Matrix<T> operator *(const Matrix<T>& larg, const Matrix<T>& rarg)
	{
		Matrix<T> dest;
		dest.resize(larg.getNRows(), rarg.getNCols());

		BlasWrapper::gemm('n', 'n', larg.getNRows(), rarg.getNCols(), larg.getNCols(), 1.0, larg.getDataPtr(), larg.getLeadingDim(), rarg.getDataPtr(),
			rarg.getLeadingDim(), 0.0, dest.getDataPtr(), dest.getLeadingDim());
		return dest;
	}

	template <typename T>
	friend Matrix<T> operator*(const Matrix<T>& larg, const T& rarg)
	{
		Matrix<T> dest(larg);
		(LABaseObject<T>&)(dest) *= rarg;
		return dest;
	}

	template <typename T>
	friend Matrix<T> operator+(const Matrix<T>& larg, const Matrix<T>&  rarg)
	{
		Matrix<T> dest(larg);
		(LABaseObject<T>&)(dest) += (const LABaseObject<T>&)rarg;
		return dest;
	}
};


