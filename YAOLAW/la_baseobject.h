#pragma once
#include <memory>
#include <string>
#include "la_operations.h"

template <typename T>
class LABaseObject
{
protected:
	struct
	{
		size_t m_leadingDim;
		size_t m_nRows;
		size_t m_nCols;
	} m_shapeInfo;

	std::unique_ptr<T[]> m_data;

	const T* getDataPtr() const { return this->m_data.get(); }
	T* getDataPtr() { return this->m_data.get(); }
	T& get(size_t nRow, size_t nCol)
	{
		if (!(nRow < this->getNRows()))
			throw std::runtime_error("failure accessing la-object's element, violated condition: row " + std::to_string(nRow) + "not in the half open range [" + std::to_string(0) + ", " + std::to_string(this->getNRows()) + ")");
		if (!(nCol < this->getNCols()))
			throw std::runtime_error("failure accessing la-object's element, violated condition: col " + std::to_string(nCol) + "not in the half open range [" + std::to_string(0) + ", " + std::to_string(this->getNCols()) + ")");

		return this->getDataPtr()[nCol * this->getLeadingDim() + nRow];
	}
	const T& get(size_t nRow, size_t nCol) const
	{
		if (!(nRow < this->getNRows()))
			throw std::runtime_error("failure accessing la-object's element, violated condition: row " + std::to_string(nRow) + "not in the half open range [" + std::to_string(0) + ", " + std::to_string(this->getNRows()) + ")");
		if (!(nCol < this->getNCols()))
			throw std::runtime_error("failure accessing la-object's element, violated condition: col " + std::to_string(nCol) + "not in the half open range [" + std::to_string(0) + ", " + std::to_string(this->getNCols()) + ")");

		return this->getDataPtr()[nCol * this->getLeadingDim() + nRow];
	}

	void copyDataTo(LABaseObject<T>& dest)
	{
		dest.resize(this.getNRows(), this.getNCols());
		BlasWrapper::copy(this.getNRows() * this.getNCols(),
			this.getDataPtr(),
			1,
			dest.getDataPtr(),
			1);
	}

	void copyDataFrom(const LABaseObject<T>& source)
	{
		this->resize(source.getNRows(), source.getNCols());
		BlasWrapper::copy(source.getNRows() * source.getNCols(),
			source.getDataPtr(),
			1,
			this->getDataPtr(),
			1);
	}

	void scale(const T& scale, LABaseObject<T>& dest)
	{
		int elem_dist = 1;
		int n = dest.getNRows() * dest.getNCols();

		BlasWrapper::scale(n, scale, dest.getDataPtr(), elem_dist);
	}


public:
	LABaseObject() : LABaseObject(0, 0) {}
	LABaseObject(const size_t nRows, size_t nCols) : m_shapeInfo(), m_data()
	{
		this->resize(nRows, nCols);
	}
	LABaseObject(const LABaseObject<T>& source) : LABaseObject()
	{
		this->copyDataFrom(source);
	}
	~LABaseObject()
	{

	}

	size_t getNRows() const
	{
		return this->m_shapeInfo.m_nRows;
	}
	size_t getNCols() const
	{
		return this->m_shapeInfo.m_nCols;
	}
	size_t getLeadingDim() const
	{
		return this->m_shapeInfo.m_leadingDim;
	}

	LABaseObject<T>& operator=(const LABaseObject<T>& src)
	{
		this->copyDataFrom(src);
		return *this;
	}

	LABaseObject<T>& operator*=(const T& value)
	{
		this->scale(value, *this);
		return *this;
	}

	LABaseObject<T>& operator+=(const T& value)
	{
		size_t nElements = this->getNRows() * this->getNCols();
		for (size_t i = 0; i < nElements; i++)
		{
			this->m_data.get()[i] += value;
		}

		return *this;
	}

	T& operator()(size_t nRow, size_t nCol)
	{
		return this->get(nRow, nCol);
	}
	const T& operator()(size_t nRow, size_t nCol) const
	{
		return this->get(nRow, nCol);
	}

	void print(std::ostream& os) const
	{
		os << "[";
		for (unsigned int r = 0; r < this->getNRows(); r++)
		{
			os << "[";
			for (unsigned int c = 0; c < this->getNCols(); c++)
			{
				(c < (this->getNCols() - 1)) ? os << (*this)(r, c) << ", " : os << (*this)(r, c);
			}
			(r < (this->getNRows() - 1)) ? os << "]\n" : os << "]";
		}
		os << "}]";
	}

	void resize(size_t nRows, size_t nCols)
	{
		/// we have col-major format so use n_cols as leading dim
		this->m_shapeInfo.m_leadingDim = nRows;
		this->m_shapeInfo.m_nRows = nRows;
		this->m_shapeInfo.m_nCols = nCols;

		size_t n_elements = this->getNRows() * this->getNCols();

		if (n_elements > 0)
		{
			this->m_data.reset(new T[n_elements]);
			for (size_t i = 0; i < n_elements; i++)
			{
				this->m_data[i] = T(0.0);
			}
		}
	}

	template <typename T>
	friend LABaseObject<T> operator+(const LABaseObject<T>& larg, const LABaseObject<T>& rarg)
	{
		if (larg->getNCols() != rarg->getNCols() || larg->getNRows() != rarg->getNRows())
			throw std::runtime_error("Shapes of the left argument and the right argument do not match.");

		LABaseObject<T> dest(larg);
		size_t nElements = larg->getNRows() * larg->getNCols();
		for (size_t i = 0; i < nElements; i++)
		{
			dest->m_data.get()[i] += rarg->m_data.get()[i];
		}

		return dest;
	}

	template <typename T>
	friend LABaseObject<T> operator*(const LABaseObject<T>& larg, const T& rarg)
	{
		LABaseObject<T> dest(larg);
		size_t nElements = larg->getNRows() * larg->getNCols();
		for (size_t i = 0; i < nElements; i++)
		{
			dest->m_data.get()[i] *=rarg;
		}

		return dest;
	}
};