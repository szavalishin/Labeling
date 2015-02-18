//Labeling Tools declaration
//Copyright (c) by Sergey Zavalishin 2010-2015
//
//See defenition in LabelingTools.hpp

#include "LabelingTools.hpp"

#include <opencv2/imgproc/imgproc.hpp>

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
		labels = cv::Mat(binImg.rows, binImg.cols, CV_32SC1);

		watch_.reset();
		watch_.start();

		DoLabel(binImg, labels, threads, coh);

		watch_.stop();

		return watch_.getTime() * 1000;
	}

	///////////////////////////////////////////////////////////////////////////////

	TImage ILabeling::RGB2Gray(const TImage& img) const
	{
		auto binImg = img;

		if (binImg.channels() > 1)
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

		auto binImg = RGB2Gray(pixels);
		labels = cv::Mat::zeros(binImg.rows, binImg.cols, CV_32SC1);

		// Initialization
		TOCLBuffer<TPixel> oclPixels(*this, TOCLBufferType::READ_ONLY, binImg.total());
		TOCLBuffer<TLabel> oclLabels(*this, TOCLBufferType::READ_WRITE, labels.total());

		memset(&oclLabels.Buffer()[0], 0, sizeof(oclLabels.Buffer()[0]) * oclLabels.Buffer().size());
		oclLabels.Push();

		memcpy(oclPixels.Buffer().data(), pixels.data, sizeof(TPixel) * binImg.total());
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

} /* LabelingTools */