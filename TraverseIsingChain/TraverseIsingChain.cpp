// TraverseIsingChain.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <complex>
#include <iostream>
#include <iomanip>
#include "../YAOLAW/la_baseobject.h"
#include "../YAOLAW/matrix.h"
#include "../YAOLAW/MatrixUtility.h"

using cdouble = std::complex<double>;

LA::Matrix<cdouble> spinX = LA::Matrix<cdouble>(2, 2);
LA::Matrix<cdouble> spinY = LA::Matrix<cdouble>(2, 2);
LA::Matrix<cdouble> spinZ = LA::Matrix<cdouble>(2, 2);

void setupSpinMatrices()
{
	spinX(0, 1) = 1;
	spinX(1, 0) = 1;

	spinX(0, 1) = cdouble(0, 1);
	spinX(1, 0) = cdouble(0, -1);

	spinX(0, 0) = 1;
	spinX(1, 1) = -1;
}

LA::Matrix<cdouble> getSpin(LA::Matrix<cdouble>& elementarySpinMatrix, size_t L, size_t i)
{
	LA::Matrix<cdouble> identity = LA::Matrix<cdouble>::identity(2);
	LA::Matrix<cdouble> result;

	if (i > 1)
	{
		result = LA::Matrix<cdouble>::identity(2);
		for (int j = 0; j < (int)i - 2; j++)
			result = result.directProduct(identity);
		result = result.directProduct(elementarySpinMatrix);
	}
	else
	{
		result = elementarySpinMatrix;
	}

	for (int j = 0; j < (int)L - (int)i; j++)
		result = result.directProduct(identity);

	return result;
}

LA::Matrix<cdouble> createHamiltonian(size_t L, double h);

LA::Matrix<cdouble> createHamiltonian(size_t L, double h)
{
	LA::Matrix<cdouble> hamilton = LA::Matrix<cdouble>((int)pow(2, L), (int)pow(2, L));

	for (size_t i = 1; i < L; i++)
	{
		LA::Matrix<cdouble> temp = getSpin(spinX, L, i);
		hamilton -= temp;
	}

	for (size_t i = 1; i < L - 1; i++)
	{
		auto A = getSpin(spinX, L, i);
		auto B = getSpin(spinX, L, i + 1);

		hamilton -= A*B;
	}

	return hamilton;
}

LA::Matrix<cdouble> createMagnetization(size_t L)
{
	LA::Matrix<cdouble> magnetization = getSpin(spinZ, L, 1);

	for (size_t i = 0; i < L; i++)
		magnetization += getSpin(spinZ, L, i);
	magnetization *= 1 / (int)L;

	return magnetization;
}



int main()
{
	setupSpinMatrices();
	size_t N = 6;
	double h = 0.5;
	LA::Matrix<cdouble> H = createHamiltonian(N, h);


	//test Lanczos method
	LA::Matrix<cdouble> T;
	LA::Vector<cdouble> v_init = LA::Vector<cdouble>(H.getNCols());
	for (size_t i = 0; i < v_init.getSize(); i++)
		v_init(i) = (double)std::rand() / RAND_MAX;

	LA::MatrixUtility::LanczosTransformation(H, T, v_init, 3);

	std::cout << "T:" << std::endl << T << std::endl;

	std::vector<cdouble> eigenvalues;
	LA::Matrix<cdouble> eigenvectors;
	H.EVD(eigenvalues, eigenvectors);

	std::cout << "eigenvalues:\n";
	for (size_t i = 0; i < eigenvalues.size(); i++)
		std::cout << eigenvalues.at(i) << ", ";
	std::cout << std::endl;

	T.EVD(eigenvalues, eigenvectors);

	std::cout << "eigenvalues of T:\n";
	for (size_t i = 0; i < eigenvalues.size(); i++)
		std::cout << eigenvalues.at(i) << ", ";
	std::cout << std::endl;

	std::vector<double> realEigenvalues;
	for (size_t i = 0; i < eigenvalues.size(); i++)
		realEigenvalues.push_back(eigenvalues.at(i).real());

	std::cout << "real eigenvalues:\n";
	for (size_t i = 0; i < realEigenvalues.size(); i++)
		std::cout << realEigenvalues.at(i) << ", ";
	std::cout << std::endl;

	double minimum = realEigenvalues.at(0);
	size_t minIndex = 0;
	for (size_t i = 1; i < realEigenvalues.size(); i++)
	{
		if (realEigenvalues.at(i) < minimum)
		{
			minimum = realEigenvalues.at(i);
			minIndex = i;
		}
	}

	std::cout << "mimal eigenvalue " << minimum << " at index " << minIndex << std::endl;

	std::cout << "min(E)/N: " << minimum / N << std::endl;

	LA::Vector<cdouble> groundstate = LA::Vector<cdouble>(2);


	return 0;
}

