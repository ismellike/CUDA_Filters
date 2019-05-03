#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include "cuda.h"

using namespace std;

#include "ImageReader.h"
#include "ImageWriter.h"

__global__ void ApplyFilter(unsigned char* data, int width, int height, int channels, float* filter, int filterSize, unsigned char* outdata)
{
	int row = threadIdx.x;
	int mid = filterSize / 2;

	//establish shared filter
	__shared__ float sharedFilter[256];

	if(row < filterSize * filterSize)
		sharedFilter[row] = filter[row];
	
	__syncthreads();

	//imagereader gives RGBA
	for (int i = 0; i < width; i++) 
	{
		//only interested in RGB
		float accumulator[4] = { 0, 0, 0, 255 };

		//filter must be odd find mid
		for(int y = -mid; y <= mid; y++)
			for (int x = -mid; x <= mid; x++) 
			{
				//check targetLoc is valid
				if (row + y < 0 || row + y >= height || i + x < 0 || i + x >= width)
				{
					//out of bounds use other values
					//CASE 1: missing pixels with value 0 so skip
					continue;
				}
				//convolution
				int targetLoc = (row - y) * width * channels + (i - x) * channels;
				int filterLoc = (y + mid) * filterSize + (x + mid);
				
				for (int channel = 0; channel < 3; channel++)
					accumulator[channel] += data[targetLoc + channel] * sharedFilter[filterLoc];
			}

		//check valid values
		for (int channel = 0; channel < channels; channel++) 
		{
			if (accumulator[channel] < 0)
				accumulator[channel] = 0;
			else if (accumulator[channel] > 255)
				accumulator[channel] = 255;

			//set outdata
			outdata[row * width * channels + i * channels + channel] = accumulator[channel];
		}
	}
}

//create array of size N^2 and return N
int ReadFilter(string filterPath, float* packedArray) 
{
	ifstream inFile(filterPath);

	if (!inFile.is_open())
	{
		cout << "Invalid Filter" << endl;
		return 0;
	}

	if (packedArray)
		delete packedArray;

	int N = 0;
	int i = 0;
	int sum = 0;

	inFile >> N;
	packedArray = new float[N * N];
	while (i < N * N) 
	{
		inFile >> packedArray[i];
		sum += packedArray[i];
		i++;
	}

	if (sum > 1) 
	{
		for(i = 0; i < N*N; i++)
			packedArray[i] /= sum;

		return N;
	}

	if (sum < 0 || (sum != 0 && sum != 1))
	{
		cout << "Invalid filter" << endl;
		return 0;
	}

	return N;
}

int main(int argc, char* argv[])
{
	if (argc < 4) {
		cout << "USAGE: project3 [picture] [filter] [output]" << endl;
		return -1;
	}
	string filePath = argv[1], filterPath = argv[2], outputPath = argv[3];

	//get filter
	float* filter = nullptr;
	int N = ReadFilter(filterPath, filter);

	//establish reader
	ImageReader* reader = ImageReader::create(filePath);
	if (reader == nullptr)
	{
		cout << "Invalid reader" << endl;
		return -1;
	}

	//read basic info
	int xRes = reader->getWidth();
	int yRes = reader->getHeight();
	int nChannels = reader->getNumChannels();

	//get picture data
	cryph::Packed3DArray<unsigned char>* packedArray = reader->getInternalPacked3DArrayImage();
	unsigned char* h_data = packedArray->getModifiableData();
	int data_count = packedArray->getTotalNumberElements();

	//send data to device
	size_t data_size = sizeof(unsigned char) * data_count;
	unsigned char* d_data;
	unsigned char* d_outdata;

	cudaMalloc((void**)&d_data, data_size);
	cudaMalloc((void**)&d_outdata, data_size);
	cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

	//send filter to device
	size_t filter_size = sizeof(float) * N * N;
	float* d_filter;

	cudaMalloc((void**)&d_filter, filter_size);
	cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice);
	//launch kernel
	ApplyFilter<<< 1, yRes >>>(d_data, xRes, yRes, nChannels, d_filter, N, d_outdata);
	cudaThreadSynchronize();

	//read kernel data
	unsigned char* h_outdata = new unsigned char[data_count];
	cudaMemcpy(h_outdata, d_outdata, data_size, cudaMemcpyDeviceToHost);

	//establish writer
	ImageWriter* writer = ImageWriter::create(outputPath, xRes, yRes, nChannels);
	if (writer == nullptr)
	{
		cout << "Invalid writer" << endl;
		return -1;
	}

	//write image
	writer->writeImage(h_outdata);
	writer->closeImageFile();

	//free memory
	delete reader, writer;
	cudaFree(d_data);
	cudaFree(d_filter);
	cudaFree(d_outdata);
	delete filter;
	delete h_outdata;
	return 0;
}