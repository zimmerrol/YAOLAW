#include "stdafx.h"
#include "matrix.h"
#include "la_wrapper.h"
#include <algorithm>

namespace LA
{
	template<typename T>
	Matrix<T>& Matrix<T>::inv()
	{
		return *this;
		// TODO: hier Rückgabeanweisung eingeben
	}

	template <typename T>
	void Matrix<T>::EVD(std::vector<std::complex<T>>& eigenvalue, Matrix<T>& eigenvectors) const
	{
		if (getNRows() != getNCols())
			throw std::runtime_error("number of row (" + std::to_string(getNRows()) + ") have to be the same as the number of columns (" + std::to_string(getNRows()) + ").");
		size_t N = getNRows();

		eigenvectors.resize(N, N);
		eigenvalue.resize(N);

		auto evReal = new T[N];
		auto evImag = new T[N];
		auto A = new T[N*N];

		BlasWrapper::copy(N*N, this->getDataPtr(), 1, A, 1);
		auto info = BlasWrapper::geev('N', 'V', N, A, this->getLeadingDim(), evReal, evImag, nullptr, this->getLeadingDim(), eigenvectors.getDataPtr(), this->getLeadingDim());

		for (size_t i = 0; i < N; i++)
			eigenvalue[i] = std::complex<T>(evReal[i], evImag[i]);

		delete[] evReal;
		delete[] evImag;
		delete[] A;
	}

	template <typename baseT>
	void Matrix<std::complex<baseT>>::EVD(std::vector<std::complex<baseT>>& eigenvalues, Matrix<std::complex<baseT>>& eigenvectors) const
	{
		if (getNRows() != getNCols())
			throw std::runtime_error("number of row (" + std::to_string(getNRows()) + ") have to be the same as the number of columns (" + std::to_string(getNRows()) + ").");
		int N = getNRows();

		eigenvectors.resize(N, N);
		eigenvalues.resize(N);
		auto A = new T[N*N];

		BlasWrapper::copy(N*N, this->getDataPtr(), 1, A, 1);
		auto info = BlasWrapper::geev('N', 'V', N, A, this->getLeadingDim(), eigenvalues.data(), nullptr, this->getLeadingDim(), eigenvectors.getDataPtr(), this->getLeadingDim());

		delete[] A;
	}

	template <typename T>
	void Matrix<T>::SVD(Matrix<T>& U, Vector<T>& Sigma, Matrix<T>& Vt) const
	{
		size_t n = this->getNRows();
		size_t m = this->getNCols();

		U.resize(n, n);
		Vt.resize(m, m);
		Sigma.resize(std::min(n, m));

		T* superb = new T[std::min(n, m) - 1];

		auto A = new T[n*m];
		BlasWrapper::copy(n*m, this->getDataPtr(), 1, A, 1);

		BlasWrapper::gesvd('A', 'A', this->getNRows(), this->getNCols(), A, this->getLeadingDim(), Sigma.getDataPtr(),
			U.getDataPtr(), U.getLeadingDim(), Vt.getDataPtr(), Vt.getLeadingDim(), superb);

		delete[] superb;
		delete[] A;
	}

	template <typename baseT>
	void Matrix<std::complex<baseT>>::SVD(Matrix<std::complex<baseT>>& U, Vector<baseT>& Sigma, Matrix<std::complex<baseT>>& Vt) const
	{
		size_t n = this->getNRows();
		size_t m = this->getNCols();

		U.resize(n, n);
		Vt.resize(m, m);
		Sigma.resize(std::min(n, m));

		baseT* superb = new baseT[std::min(n, m) - 1];

		auto A = new std::complex<baseT>[n*m];
		BlasWrapper::copy(n*m, this->getDataPtr(), 1, A, 1);

		BlasWrapper::gesvd('A', 'A', this->getNRows(), this->getNCols(), A, this->getLeadingDim(), Sigma.getDataPtr(),
			U.getDataPtr(), U.getLeadingDim(), Vt.getDataPtr(), Vt.getLeadingDim(), superb);

		delete[] superb;
		delete[] A;
	}

	template <typename T>
	Matrix<T> Matrix<T>::transpose()
	{
		Matrix<T> result = Matrix<T>(this->getNRows(), this->getNCols());
		BlasWrapper::omatcopy('T', this->getNRows(), this->getNCols(), (T)1.0, this->getDataPtr(), this->getNRows(), result.getDataPtr(), this->getNCols());
		result.resize(this->getNCols(), this->getNRows());
		return result;
	}

	template <typename baseT>
	Matrix<std::complex<baseT>> Matrix<std::complex<baseT>>::transpose()
	{
		Matrix<std::complex<baseT>> result = Matrix<std::complex<baseT>>(this->getNRows(), this->getNCols());
		BlasWrapper::omatcopy('T', this->getNRows(), this->getNCols(), (T)1.0, this->getDataPtr(), this->getNRows(), result.getDataPtr(), this->getNCols());
		result.resize(this->getNCols(), this->getNRows());
		return result;
	}

	template <typename baseT>
	Matrix<std::complex<baseT>> Matrix<std::complex<baseT>>::adjungate()
	{
		Matrix<std::complex<baseT>> result = this->transpose();
		size_t nElements = this->getNCols() * this->getNRows();
		for (size_t i = 0; i < nElements; i++)
		{
			result.getDataPtr()[i] = std::conj(result.getDataPtr()[i]);
		}

		return result;
	}

	template <typename T>
	Matrix<T> Matrix<T>::directProduct(const Matrix<T>& rarg) const
	{
		Matrix<T> result = Matrix<T>(getNRows()*rarg.getNRows(), getNCols()*rarg.getNCols());

		for (size_t n = 0; n < getNRows(); n++)
			for (size_t m = 0; m < getNCols(); m++)
				for (size_t i = 0; i < rarg.getNRows(); i++)
					for (size_t j = 0; j < rarg.getNCols(); j++)
						result(i*getNRows() + n, j*getNCols() + m) = get(n, m) * rarg(i, j);

		return result;
	}

	template <typename baseT>
	Matrix<std::complex<baseT>> Matrix<std::complex<baseT>>::directProduct(const Matrix<std::complex<baseT>>& rarg) const
	{
		Matrix<std::complex<baseT>> result = Matrix<std::complex<baseT>>(getNRows()*rarg.getNRows(), getNCols()*rarg.getNCols());

		for (size_t n = 0; n < getNRows(); n++)
			for (size_t m = 0; m < getNCols(); m++)
				for (size_t i = 0; i < rarg.getNRows(); i++)
					for (size_t j = 0; j < rarg.getNCols(); j++)
						result(i*getNRows() + n, j*getNCols() + m) = get(n, m) * rarg(i, j);

		return result;
	}

	template<typename T>
	Matrix<T> Matrix<T>::identity(size_t n)
	{
		Matrix<T> result = Matrix<T>(n, n);
		for (size_t i = 0; i < n; i++)
			result(i, i) = 1.0;

		return result;
	}

	template<typename baseT>
	Matrix<std::complex<baseT>> Matrix<std::complex<baseT>>::identity(size_t n)
	{
		Matrix<std::complex<baseT>> result = Matrix<std::complex<baseT>>(n, n);
		for (size_t i = 0; i < n; i++)
			result(i, i) = 1.0;

		return result;
	}

	template class Matrix<double>;
	template class Matrix<float>;
	template class Matrix<std::complex<double>>;
	template class Matrix<std::complex<float>>;
}