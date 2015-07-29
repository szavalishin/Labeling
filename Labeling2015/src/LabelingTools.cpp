//Labeling Tools declaration
//Copyright (c) by Sergey Zavalishin 2010-2015
//
//See defenition in LabelingTools.hpp

#include "LabelingTools.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <array>

///////////////////////////////////////////////////////////////////////////////

namespace LabelingTools
{

	///////////////////////////////////////////////////////////////////////////////
	// ILabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	TTime ILabeling::Label(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		THROW_IF(pixels.empty(), "ILabeling::Label : Input image is empty");

		TImage binImg = RGB2Gray(pixels);
		labels = cv::Mat::zeros(binImg.rows, binImg.cols, CV_32SC1);

		watch_.reset();
		watch_.start();
		
		DoLabel(binImg, labels, threads, coh);

		watch_.stop();

		return watch_.getTime() * 1000;
	}

	///////////////////////////////////////////////////////////////////////////////

	TImage ILabeling::RGB2Gray(const TImage& img) const
	{
		TImage binImg = img;

		if (img.channels() > 1)
			cv::cvtColor(binImg, binImg, cv::COLOR_RGB2GRAY);
		
		cv::threshold(binImg, binImg, cv::THRESH_OTSU, 255, CV_8UC1);
		
		return binImg;
	}

	///////////////////////////////////////////////////////////////////////////////

	void ILabeling::SetupThreads(char threadCount)
	{
		if (threadCount != MAX_THREADS)
		{
			omp_set_dynamic(0);
			omp_set_num_threads(threadCount);
		}
		else
		{
			omp_set_dynamic(1);
			omp_set_num_threads(omp_get_max_threads());
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	// IOCLLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	IOCLLabeling::IOCLLabeling(void)
		: isInitialized(false),
		  Initialized(isInitialized),
		  State(OCLState)
	{
		/* Empty */
	}

	///////////////////////////////////////////////////////////////////////////////

	void IOCLLabeling::Init(cl_device_type deviceType, const std::string& buildParams, const std::string& srcFileName)
	{
		TerminateOCL();

		clInitParams params = { deviceType, "", "" };
		strcpy_s(params.build_params, buildParams.c_str());
		strcpy_s(params.kernel_source_file_name, srcFileName.c_str());

		int err = InitOpenCL(&OCLState, &params);
		isInitialized = !err;
		THROW_IF_OCL(err, "IOCLLabeling::Init::InitOpenCL");

		InitKernels();		
	}

	///////////////////////////////////////////////////////////////////////////////

	TTime IOCLLabeling::Label(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		THROW_IF(!Initialized, "IOCLLabeling::Label : OpenCL device is not initialized");
		THROW_IF(pixels.empty(), "IOCLLabeling::Label : Input image is empty");

		auto binImg = cv::Mat(cv::Size((pixels.cols >> 5 << 5) + 32, (pixels.rows >> 5 << 5) + 32), CV_8UC1);
		RGB2Gray(pixels).copyTo(binImg(cv::Rect(0, 0, pixels.cols, pixels.rows)));
		
		labels = cv::Mat::zeros(binImg.rows, binImg.cols, CV_32SC1);

		// Initialization
		TOCLBuffer<TPixel> oclPixels(*this, TOCLBufferType::READ_ONLY, binImg.total());
		TOCLBuffer<TLabel> oclLabels(*this, TOCLBufferType::READ_WRITE, labels.total());

		memset(&oclLabels.Buffer()[0], 0, sizeof(oclLabels.Buffer()[0]) * oclLabels.Buffer().size());
		oclLabels.Push();
		
		memcpy(oclPixels.Buffer().data(), binImg.data, sizeof(TPixel) * binImg.total());
		oclPixels.Push();
		
		watch_.reset();
		watch_.start();

		// Actual Code
		DoOCLLabel(oclPixels, oclLabels, labels.cols, labels.rows, coh);

		// Post Conditions
		watch_.stop();

		oclLabels.Pull();
		memcpy(labels.data, oclLabels.Buffer().data(), sizeof(TLabel) * labels.total());
		
		return watch_.getTime() * 1000;
	}

	///////////////////////////////////////////////////////////////////////////////

	void IOCLLabeling::TerminateOCL(void)
	{
		int err = CL_SUCCESS;

		if (isInitialized)
		{
			FreeKernels();
			err = TerminateOpenCL(&OCLState);
		}

		THROW_IF_OCL(err, "IOCLLabeling::TerminateOCL");
	}

	///////////////////////////////////////////////////////////////////////////////

	IOCLLabeling::~IOCLLabeling(void)
	{
		TerminateOCL();
	}

	///////////////////////////////////////////////////////////////////////////////
	// IOCLLabeling3D declaration
	///////////////////////////////////////////////////////////////////////////////

	TTime IOCLLabeling3D::Label(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		THROW_IF(!Initialized, "IOCLLabeling3D::Label : OpenCL device is not initialized");
		THROW_IF(pixels.empty(), "IOCLLabeling3D::Label : Input image is empty");
		THROW_IF(pixels.dims != 3, "IOCLLabeling3D::Label : Input image is not a 3D image");
		THROW_IF(coh != TCoherence::COH_DEFAULT, "IOCLLabeling3D::Label : Only default coherence is supported for 3D labeling");

		int sz[3]; 

		PlaneIterator itPix(pixels);

		for (int i = 0; i < pixels.dims; ++i)
			sz[i] = (pixels.size[i] >> 5 << 5) + 32; // 32 is NVidia specific (try 64 for AMD)
		
		auto binImg = TImage(3, sz, CV_8U, cv::Scalar(0));
		PlaneIterator itBin(binImg);	

		for (int i = 0; i < itPix.NPlanes(); ++i)
		{
			const TImage pixPlane = itPix.Plane();
			TImage binPlane = itBin.Plane();

			RGB2Gray(pixPlane).copyTo(binPlane(cv::Rect(0, 0, pixPlane.rows, pixPlane.cols)));

			++itPix;
			++itBin;
		}

		labels = cv::Mat::zeros(3, sz, CV_32SC1);

		// Initialization
		TOCLBuffer<TPixel> oclPixels(*this, TOCLBufferType::READ_ONLY, binImg.total());
		TOCLBuffer<TLabel> oclLabels(*this, TOCLBufferType::READ_WRITE, labels.total());

		memset(&oclLabels.Buffer()[0], 0, sizeof(oclLabels.Buffer()[0]) * oclLabels.Buffer().size());
		oclLabels.Push();

		memcpy(oclPixels.Buffer().data(), binImg.data, sizeof(TPixel) * binImg.total());
		oclPixels.Push();

		watch_.reset();
		watch_.start();

		// Actual Code
		DoOCLLabel3D(oclPixels, oclLabels, labels.size[0], labels.size[1], labels.size[2]);

		// Post Conditions
		watch_.stop();

		oclLabels.Pull();
		memcpy(labels.data, oclLabels.Buffer().data(), sizeof(TLabel) * labels.total());

		return watch_.getTime() * 1000;
	}

	///////////////////////////////////////////////////////////////////////////////

} /* LabelingTools */