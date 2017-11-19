#pragma once
#include <stdlib.h>
#include "complex"

namespace LA
{
	namespace MathUtility
	{
		bool approximateEqual(double v1, double v2, double epsilon = 1e-5)
		{
			return std::fabs(v1 - v2) < epsilon;
		}

		bool approximateEqual(float v1, float v2, float epsilon = 1e-5)
		{
			return std::fabs(v1 - v2) < epsilon;
		}

		template <typename T>
		bool approximateEqual(std::complex<T> v1, std::complex<T> v2, T epsilon = 1e-5)
		{
			return std::abs(v1 - v2) < epsilon;
		}
	}
}