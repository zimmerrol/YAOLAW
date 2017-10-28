#include "stdafx.h"
#include "matrix.h"
#include "la_wrapper.h"

template <typename T>
Matrix<T>& Matrix<T>::operator *=(const T& value)
{
	(LABaseObject<T>&)(*this) *= value;
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::inv()
{
	// TODO: hier Rückgabeanweisung eingeben
}

/*
template <typename T>
Matrix<T> operator*(const Matrix<T>& rhs)
{
	result = Matrix<T>(this);
	result *= right;
	return result;
}*/