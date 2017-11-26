#pragma once

#include "matrix.h"

namespace LA
{
	namespace MPSUtility
	{
		template <typename std::complex<BaseT>>
		Matrix<std::complex<BaseT>> orthogonalize(Matrix<std::complex<BaseT>> M1, Matrix<std::complex<BaseT>> M2, bool performLeftOrthogonalization = true)
		{
			using T = std::complex<baseT>;

			if (M1.getNCols() != M2.getNCols())
				throw std::runtime_error("");
			if (M1.getNRows() != M2.getNRows())
				throw std::runtime_error("");

			Matrix<T> M;

			Matrix<T> A;
			Matrix<T> S;
			Matrix<T> V;

			if (performLeftOrthogonalization)
			{
				M = Matrix<T>(M1.getNRows() + M2.getNRows(), M1.getNCols());

				for (size_t n = 0; n < M1.getNRows(); n++)
				{
					for (size_t m = 0; n < M1.getNCols(); m++)
					{
						M(n, m) = M1(n, m);
					}
				}

				for (size_t n = 0; n < M2.getNRows(); n++)
				{
					for (size_t m = 0; n < M2.getNCols(); m++)
					{
						M(n + M1.getNRows(), m) = M2(n, m);
					}
				}
			}
			else
			{
				M = Matrix<T>(M1.getNRows(), M1.getNCols() + M2.getNCols());

				for (size_t n = 0; n < M1.getNRows(); n++)
				{
					for (size_t m = 0; n < M1.getNCols(); m++)
					{
						M(n, m) = M1(n, m);
					}
				}

				for (size_t n = 0; n < M2.getNRows(); n++)
				{
					for (size_t m = 0; n < M2.getNCols(); m++)
					{
						M(n, m + M1.getNCols()) = M2(n, m);
					}
				}

				M = M.adjungate();
			}

			M.SVD(A, S, V);

			if (performLeftOrthogonalization)
			{

			}
			else
			{

			}
		}
	}
}