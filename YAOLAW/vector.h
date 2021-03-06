#pragma once
#include "la_baseobject.h"
#include <algorithm>
#include <vector>

namespace LA
{
	template <typename T>
	struct getInnerType
	{
		using type = T;
	};

	template <typename T>
	struct getInnerType<std::complex<T>>
	{
		using type = T;
	};

	template <typename T>
	class Matrix;

	template <typename T>
	class Vector :
		public LABaseObject<T>
	{
	protected:
		bool m_isColumnVector;
	public:
		using innerT = typename getInnerType<T>::type;

		Vector(const size_t n, bool columnVector = true) : LABaseObject<T>(columnVector ? n : 1, columnVector ? 1 : n), m_isColumnVector(columnVector) {}
		Vector() : LABaseObject<T>(), m_isColumnVector(false) {}
		Vector(const Vector<T>& _other) : LABaseObject<T>(_other), m_isColumnVector(_other.m_isColumnVector) {}

		Vector(std::vector<T> value, bool isColumnVector = true) : this(value.size(), isColumnVector)
		{
			for (size_t i = 0; i < value.size(); i++)
			{
				this->operator()(i) = value[i];
			}
		}


		size_t getSize() const
		{
			return std::max(getNRows(), getNCols());
		}

		Vector<T> transpose() const
		{
			auto result = Vector<T>(this->getSize(), !this->m_isColumnVector);
			result.copyDataFrom(*this);
			return result;
		}

		innerT norm()
		{
			size_t size = getSize();
			innerT result = 0.0;
			for (size_t i = 0; i < size; i++)
			{
				result += std::norm(this->operator()(i));
			}
			return std::sqrt(result);
		}

		void resize(size_t n)
		{
			if (m_isColumnVector)
			{
				LABaseObject::resize(n, 1);
			}
			else
			{
				LABaseObject::resize(1, n);
			}
		}

		T& operator()(size_t n)
		{
			return this->get(m_isColumnVector ? n : 0, m_isColumnVector ? 0 : n);
		}
		const T& operator()(size_t n) const
		{
			return this->get(m_isColumnVector ? n : 0, m_isColumnVector ? 0 : n);
		}

		friend T operator*(const Vector<T>& larg, const Vector<T>& rarg)
		{
			if (larg.m_isColumnVector || !rarg.m_isColumnVector)
				throw std::runtime_error("You have to multiply a row wise with a column wise vector.");

			int n = larg.getSize();
			if (n != rarg.getSize())
				throw std::runtime_error("Size of the two vectors does not match: " + std::to_string(n) + " vs " + std::to_string(rarg.getSize()) + ".");

			return BlasWrapper::dot(n, larg.getDataPtr(), 1, rarg.getDataPtr(), 1);
		}

		friend Vector<T> operator*(const Matrix<T>& larg, const Vector<T>& rarg)
		{
			if (!rarg.m_isColumnVector)
				throw std::runtime_error("Vector has to be column wise.");
			if (larg.getNCols() != rarg.getSize())
				throw std::runtime_error("Number of matrix columns (" + std::to_string(larg.getNCols()) + ") does not match the size of the vector (" + std::to_string(rarg.getSize()) + ").");
			auto result = Vector<T>(larg.getNRows());
			BlasWrapper::gemv('n', larg.getNRows(), larg.getNCols(), (T)1.0, larg.getDataPtr(), larg.getLeadingDim(), rarg.getDataPtr(), 1, (T)0.0, result.getDataPtr(), 1);
			return result;
		}

		//TODO: fix!
		friend Vector<T> operator*(const Vector<T>& larg, const Matrix<T>& rarg)
		{
			if (larg.m_isColumnVector)
				throw std::runtime_error("Vector has to be row wise.");
			if (larg.getNCols() != rarg.getNRows())
				throw std::runtime_error("Vector size (" + std::to_string(larg.getSize()) + ") does not match the number of columns of the matrix (" + std::to_string(rarg.getNCols()) + ").");
			auto result = Vector<T>(larg.getNCols(), false);
			result -= 1;
			BlasWrapper::gemm('n', 'n', 1, rarg.getNCols(), rarg.getNRows(), (T)1.0, larg.getDataPtr(), larg.getLeadingDim(), rarg.getDataPtr(), rarg.getNRows(), (T)0.0, result.getDataPtr(), result.getLeadingDim());

			return result;
		}
		

		friend Vector<T> operator+(const Vector<T>& larg, const Vector<T>&  rarg)
		{
			Vector<T> dest(larg);
			(LABaseObject<T>&)(dest) += (const LABaseObject<T>&)rarg;
			return dest;
		}

		friend Vector<T> operator-(const Vector<T>& larg, const Vector<T>&  rarg)
		{
			Vector<T> dest(larg);
			(LABaseObject<T>&)(dest) -= (const LABaseObject<T>&)rarg;
			return dest;
		}

		friend Vector<T> operator*(const Vector<T>& larg, const T& rarg)
		{
			Vector<T> dest(larg);
			(LABaseObject<T>&)(dest) *= rarg;
			return dest;
		}

		/*friend Vector<T> operator*(const T& larg, const Vector<T>& rarg)
		{
			Vector<T> dest(larg);
			(LABaseObject<T>&)(dest) *= rarg;
			return dest;
		}*/

		friend Vector<T> operator/(const Vector<T>& larg, const T& rarg)
		{
			Vector<T> dest(larg);
			(LABaseObject<T>&)(dest) /= rarg;
			return dest;
		}

		/*	friend Vector<T> operator/(const T& larg, const Vector<T>& rarg)
			{
				Vector<T> dest(larg);
				(LABaseObject<T>&)(dest) /= rarg;
				return dest;
			}*/
	};
}