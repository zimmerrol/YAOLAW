#pragma once
#include "matrix.h"
#include "vector.h"
#include "MathUtility.h"

namespace LA
{
	namespace MatrixUtility
	{
		template <typename T_>
		void LanczosTransformation(Matrix<std::complex<T_>>& H, Matrix<std::complex<T_>>& T, LA::Vector<std::complex<T_>> v, size_t m)
		{
			//normalize the input vector
			double vNorm = v.norm();
			if (!LA::MathUtility::approximateEqual<double>(vNorm, 1.0))
			{
				v /= vNorm;
			}

			if (T.getNCols() != m || T.getNRows() != m)
			{
				T.resize(m, m);
			}
			//T = (std::complex<T_>)0.0;

			Vector<std::complex<T_>> w_prime;
			Vector<std::complex<T_>> w;
			Vector<std::complex<T_>> v_old;

			//initial iteration step
			w_prime = H * v;
			std::complex<T_> alpha = w_prime.transpose() * v;
			w = w_prime - v*alpha; //

			T(0, 0) = alpha;


			for (size_t j = 1; j < m; j++)
			{
				//TODO: check if this is okay...
				v_old = v;

				T_ beta = w.norm();
				if (!LA::MathUtility::approximateEqual(beta, 0, 1e-15))
				{
					v = w / beta;
				}
				else
				{
					std::cout << "WARNING: beta has vanished!" << std::endl;
					std::cout << "Changing m from " << m << " to " << j << std::endl;
					T.resize(j, j);

					return;

					//TODO:
					//pick as {\displaystyle v_{j}} v_{j} an arbitrary vector with Euclidean norm {\displaystyle 1} 1 that is orthogonal
					//to all of {\displaystyle v_{1},\dots ,v_{j-1}} {\displaystyle v_{1},\dots ,v_{j-1}}
				}

				w_prime = H * v;
				alpha = w_prime.transpose() * v;
				w = w_prime - v * alpha -  v_old * ((std::complex<T_>)beta); // 

				T(j, j - 1) = beta;
				T(j - 1, j) = beta;
				T(j, j) = alpha;
			}
		}
	}
}