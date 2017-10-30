#pragma once
#include "stdafx.h"

namespace LA
{
	class ShapeInfo
	{
	protected:
		size_t m_leadingDim;
		size_t m_nRows;
		size_t m_nCols;
	public:
		ShapeInfo() : m_leadingDim(-1), m_nRows(-1), m_nCols(-1) {}
		ShapeInfo(size_t leadingDim, size_t nRows, size_t nCols) : m_leadingDim(leadingDim), m_nRows(nRows), m_nCols(nCols) {}

		size_t getLeadingDim() const { return m_leadingDim; }
		size_t getNRows() const { return m_nRows; }
		size_t getNCols() const { return m_nCols; }
	};
}