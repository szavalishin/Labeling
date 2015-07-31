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

	TImage ILabeling::RGB2Gray(const TImage& img)
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
		labels = labels(cv::Rect(0, 0, pixels.cols, pixels.rows));
		
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

	template <typename CPP_TYPE, int CV_TYPE>
		TImage IOCLLabeling3D::CopyAlignImg(const TImage &im, uchar padding, uchar align) const
		{
			int sz[3];
			for (int i = 0; i < im.dims; ++i)
				sz[i] = ((im.size[i] + padding * 2) >> align << align) + (2 << align);

			auto outIm = TImage(3, sz, CV_TYPE, cv::Scalar(0));

			const int shift = padding >> 1;
			for (int k = 0; k < im.size[2]; ++k)
				for (int j = 0; j < im.size[1]; ++j)
					for (int i = 0; i < im.size[0]; ++i)
						outIm.at<CPP_TYPE>(i + padding, j + padding, k + padding) = im.at<CPP_TYPE>(i, j, k) > 128;

			return outIm;
		}

	///////////////////////////////////////////////////////////////////////////////

	TTime IOCLLabeling3D::Label(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		THROW_IF(!Initialized, "IOCLLabeling3D::Label : OpenCL device is not initialized");
		THROW_IF(pixels.empty(), "IOCLLabeling3D::Label : Input image is empty");
		THROW_IF(pixels.dims != 3, "IOCLLabeling3D::Label : Input image is not a 3D image");
		THROW_IF(coh != TCoherence::COH_DEFAULT, "IOCLLabeling3D::Label : Only default coherence is supported for 3D labeling");

		const uchar padding = 2;
		auto log2i = [](uchar x)
		{
			uchar r = 0; 
			while (x >>= 1) ++r; 
			return r;
		};
		
		TImage binImg = CopyAlignImg<uchar, CV_8U>(pixels, padding, log2i(imAlign));
		labels = cv::Mat::zeros(3, binImg.size, CV_32SC1);

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