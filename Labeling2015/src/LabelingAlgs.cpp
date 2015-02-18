// Labeling Alhorithms declaration
// Copyright (c) by Sergey Zavalishin 2010-2015
//
// See defenition in LabelingAlgs.hpp

#include "LabelingAlgs.hpp"
#include "cvlabeling_imagelab.h"

#include <limits>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>

//#pragma optimize("", off);

///////////////////////////////////////////////////////////////////////////////

namespace LabelingTools
{

	///////////////////////////////////////////////////////////////////////////////
	// TBinLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	void TBinLabeling::DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		SetupThreads(threads);

#		pragma omp parallel for
		for (int i = 0; i < pixels.total(); ++i)
		{
			if (pixels.data[i])
				labels.at<TLabel>(i) = 1;
			else
				labels.at<TLabel>(i) = 0;
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	// TOpenCVLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	void TOpenCVLabeling::DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		cv::connectedComponents(pixels, labels, coh == COH_8 ? 8 : 4, CV_32SC1);
	}

	///////////////////////////////////////////////////////////////////////////////
	// TBlockGranaLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	void TBlockGranaLabeling::DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		int numLabels;
		cvLabelingImageLab(&IplImage(pixels), &IplImage(labels), 255, &numLabels);
	}

	///////////////////////////////////////////////////////////////////////////////
	// TRunLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	TRunLabeling::TRunLabeling(void)
	{
		Top = 0;
		Bottom = 0;
	}

	///////////////////////////////////////////////////////////////////////////////

	TRunLabeling::TRunLabeling(unsigned int aTop, unsigned int aBottom)
	{
		Top = aTop;
		Bottom = aBottom;
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunLabeling::SetRunLabel(void)
	{
		CurRun->Label = UINT_MAX; //setting up maximum value for current label
		LabelStack.clear();

		//for all runs in previous row
		for (int i = 0; i < LastRow.size(); i++)
		{
			//if run is connected to current
			if ((int)LastRow[i]->r >= (int)CurRun->l - ConPix &&
				LastRow[i]->l <= CurRun->r + ConPix)
			{
				//Getting it's provisional label
				//if provisional label is smaller than current run label, we're assigning it to 
				//current run and adding to stack
				if (Objects[LastRow[i]->Label] < CurRun->Label)
				{
					CurRun->Label = Objects[LastRow[i]->Label];
					LabelStack.push_back(CurRun->Label);
				}
				else
				{
					//if not then we're adding provisional label to stack
					LabelStack.push_back(Objects[LastRow[i]->Label]);
				}
			}
		}

		//if there's no connected runs, we're assigning new label
		if (CurRun->Label == UINT_MAX)
		{
			Objects.push_back(Objects.size());
			CurRun->Label = Objects.size() - 1;
		}
		else
		{
			//if there are then we're reassigning all provisional labels to current
//#			pragma omp parallel for
			for (int i = 1; i < Objects.size(); ++i)
			{
				for (int j = 0; j < LabelStack.size(); ++j)
				{
					if (Objects[i] == LabelStack[j])
					{
						Objects[i] = CurRun->Label;
						continue;
					}
				}
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunLabeling::DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		//Setting thread count
		if (Bottom == 0)
			SetupThreads(threads);

		ConPix = int(coh == COH_8); //represents if we need additional 
									//pixels at left and right due to the 8x coherence

		Objects.reserve((pixels.rows * pixels.cols) >> 2);
		Runs.reserve(Objects.size());
		LastRow.reserve(pixels.cols >> 1);
		CurRow.reserve(LastRow.size());
		LabelStack.reserve(pixels.rows);

		CurRun = 0; //current run
		Objects.push_back(0);
		if (Top > Bottom || Bottom > labels.rows - 1 || Bottom == 0)
		{
			Top = 0;
			Bottom = labels.rows - 1;
		}

		//Main cycle
		for (unsigned int Row = Top; Row < Bottom; ++Row)
		{
			//scanning pixels in current row
			for (unsigned int i = 0; i < labels.cols; ++i)
			{
				//if current pixel is black or it's row end
				if (!pixels.data[Row * labels.cols + i] || i == labels.cols - 1)
				{
					//if we're in run
					if (i > 0)
						if (CurRun != 0)
						{
							//saving current run end
							CurRun->r = i - 1;
							//checking run label
							SetRunLabel();
							CurRun = 0;
						}
				}
				else
				{ //if it's white
					//if it's first pixel in run
					if (CurRun == 0)
					{
						//creating new run
						CurRun = new TRun;
						CurRun->l = i;
						CurRun->Row = Row;
						Runs.push_back(CurRun);
						CurRow.push_back(CurRun);
					}
				}
			}
			//moving to next row
			LastRow.clear();
			LastRow = CurRow;
			CurRow.clear();
		}

		//setting up labels
//#		pragma omp parallel for
		for (long int i = 0; i < Runs.size(); ++i)
		{
			for (long int j = Runs[i]->l; j <= Runs[i]->r; ++j)
			{
				labels.at<TLabel>(Runs[i]->Row, j) = Objects[Runs[i]->Label];
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	// TLabelDistribution declaration
	///////////////////////////////////////////////////////////////////////////////

	void TLabelDistribution::DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		SetupThreads(threads);

		InitMap(pixels, labels);

		bool noChanges;
		do
		{
			noChanges = Scan(labels);
			Analyze(labels);
		} while (!noChanges);
	}

	///////////////////////////////////////////////////////////////////////////////

	void TLabelDistribution::InitMap(const TImage& pixels, TImage& labels)
	{
		labels = TImage(pixels.rows, pixels.cols, CV_32SC1);

		TLabel *lb = reinterpret_cast<TLabel*>(labels.data);
		TPixel *px = reinterpret_cast<TPixel*>(pixels.data);

		#pragma omp parallel for
		for (long int i = 0; i < labels.total(); ++i)
		{
			lb[i] = px[i] ? i : 0;
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	TLabel TLabelDistribution::MinLabel(TLabel lb1, TLabel lb2) const
	{
		if (lb1 && lb2)
			return min(lb1, lb2);

		TLabel lb = max(lb1, lb2);

		return lb ? lb : UINT_MAX;
	}

	///////////////////////////////////////////////////////////////////////////////

	TLabel TLabelDistribution::MinNWSELabel(const TLabel* lb, uint lbPos, uint width, uint maxPos) const
	{		
		long int pos = lbPos;

		TLabel 
			lb1 = pos - 1     > -1     ? lb[pos - 1] : 0,
			lb2 = pos + 1     < maxPos ? lb[pos + 1] : 0,
			lb3 = pos - width > -1     ? lb[pos - width] : 0,
			lb4 = pos + width < maxPos ? lb[pos + width] : 0;

		lb1 = MinLabel(lb1, lb2);
		lb2 = MinLabel(lb3, lb4);

		return MinLabel(lb1, lb2);
	}
	
	///////////////////////////////////////////////////////////////////////////////
	
	TLabel TLabelDistribution::GetLabel(const TLabel* labels, uint pos, uint maxPos) const
	{		
		return pos && pos < maxPos ? labels[pos] : 0;
	}

	///////////////////////////////////////////////////////////////////////////////

	bool TLabelDistribution::Scan(TImage& labels)
	{
		bool noChanges = true;

		TLabel *lb = reinterpret_cast<TLabel*>(labels.data);

		#pragma omp parallel for
		for (long int i = 0; i < labels.total(); ++i)
		{
			TLabel label = lb[i];

			if (label)
			{
				TLabel minLabel = MinNWSELabel(lb, i, labels.cols, labels.total());

				if (minLabel < label)
				{					
					lb[label] = min(lb[label], minLabel);
					noChanges = false;
				}
			}
		}

		return noChanges;
	}

	///////////////////////////////////////////////////////////////////////////////

	void TLabelDistribution::Analyze(TImage& labels)
	{
		TLabel *lb = reinterpret_cast<TLabel*>(labels.data);

		#pragma omp parallel for
		for (long int i = 0; i < labels.total(); ++i)
		{
			TLabel label = lb[i];

			if (label)
			{
				TLabel curLabel = lb[label];
				while (curLabel != label)
				{
					label = lb[curLabel];
					curLabel = lb[label];
				}

				lb[i] = label;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	// TRunEqivLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		SetupThreads(threads);

		pixels_ = &pixels;
		labels_ = &labels;
		width_ = pixels_->cols;
		height_ = pixels_->rows;

		InitRuns();
		FindRuns();
		FindNeibRuns();
		Scan();
		SetFinalLabels();

		runs_.clear();
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::InitRuns(void)
	{
		runs_.resize(height_ * (width_ >> 1), { 0, 0, 0, { 1, 0 }, {1, 0} });
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::FindRuns(void)
	{		
		#pragma omp parallel for
		for (int row = 0; row < height_; ++row)
		{
			uint pixPos = row * width_;
			uint rowPos = row * (width_ >> 1);

			TPixel *curPix = reinterpret_cast<TPixel*>(pixels_->data) + pixPos;
			TRun *curRun = runs_.data() + rowPos;

			for (uint pos = 0, runPos = 0; pos < width_; )
			{
				if (*curPix) {
					if (!curRun->lb) {
						curRun->lb = rowPos + ++runPos;
						curRun->l = pos;
					}
					if (pos == width_ - 1) {
						curRun->r = pos;
					}
				}
				else {
					if (curRun->lb) {
						curRun->r = pos - 1;
						++curRun;
					}
				}
				++curPix;
				++pos;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	bool TRunEqivLabeling::IsNeib(const TRun *r1, const TRun *r2) const
	{
		return
			(r1->l >= r2->l && r1->l <= r2->r) ||
			(r1->r >= r2->l && r1->r <= r2->r) ||
			(r2->r >= r1->l && r2->r <= r1->r) ||
			(r2->r >= r1->l && r2->r <= r1->r);
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::FindNeibRuns(TRun *curRun, TRunSize *neibSize, const TRun *neibRow, uint *neibPos, uint runWidth)
	{
		int noNeib = 0;

		while (*neibPos < runWidth && !noNeib)
		{
			if (IsNeib(curRun, neibRow + *neibPos)) {
				if (neibSize->l > neibSize->r) {
					neibSize->l = neibRow[*neibPos].lb - 1;
				}
				neibSize->r = neibRow[*neibPos].lb - 1;

				if (*neibPos + 1 < runWidth   &&
					neibRow[*neibPos + 1].lb &&
					neibRow[*neibPos + 1].l <= curRun->r)
				{
					++(*neibPos);
				}
				else {
					noNeib = 1;
				}
			}
			else {
				if (neibRow[*neibPos].r < curRun->l) {
					++(*neibPos);
				}
				else {
					noNeib = 1;
				}
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::FindNeibRuns(void)
	{		
		uint runWidth = width_ >> 1;
		TRun *runs = runs_.data();

		#pragma omp parallel for
		for (int row = 0; row < height_; ++row)
		{
			TRun *curRun = runs + row * runWidth;
			const TRun *topRow = curRun - runWidth;
			const TRun *botRow = curRun + runWidth;

			uint topPos = 0;
			uint botPos = 0;

			for (uint pos = 0; curRun->lb != 0 && pos < runWidth; ++pos)
			{
				if (row > 0) {					
					FindNeibRuns(curRun, &curRun->top, topRow, &topPos, runWidth);
				}

				if (row < height_ - 1) {					
					FindNeibRuns(curRun, &curRun->bot, botRow, &botPos, runWidth);
				}

				++curRun;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	TLabel TRunEqivLabeling::MinRunLabel(uint pos)
	{
		TLabel minLabel = UINT_MAX;
		TRun *runs = runs_.data();
		TRun curRun = runs[pos];

		if (curRun.top.l <= curRun.top.r) {
			for (uint i = curRun.top.l; i < curRun.top.r + 1; ++i) {
				minLabel = min(minLabel, runs[i].lb);
			}
		}

		if (curRun.bot.l <= curRun.bot.r) {
			for (uint i = curRun.bot.l; i < curRun.bot.r + 1; ++i) {
				minLabel = min(minLabel, runs[i].lb);
			}
		}

		return minLabel;
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::Scan(void)
	{
		bool noChanges;
		do
		{
			noChanges = ScanRuns();
			AnalyzeRuns();
		} while (!noChanges);
	}

	///////////////////////////////////////////////////////////////////////////////

	bool TRunEqivLabeling::ScanRuns(void)
	{
		bool noChanges = true;
		TRun *runs = runs_.data();

		#pragma omp parallel for
		for (int pos = 0; pos < runs_.size(); ++pos)
		{
			uint runWidth = width_ >> 1;
			TLabel label = runs[pos].lb;

			if (label)
			{
				TLabel minLabel = MinRunLabel(pos);

				if (minLabel < label)
				{
					TLabel tmpLabel = runs[label - 1].lb;
					runs[label - 1].lb = min(tmpLabel, minLabel);
					noChanges = false;
				}
			}
		}

		return noChanges;
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::AnalyzeRuns(void)
	{
		TRun *runs = runs_.data();

		#pragma omp parallel for
		for (int pos = 0; pos < runs_.size(); ++pos)
		{
			TRun *curRun = &runs[pos];
			TLabel label = curRun->lb;			

			if (label){
				TLabel curLabel = runs[label - 1].lb;
				while (curLabel != label)
				{
					label = runs[curLabel - 1].lb;
					curLabel = runs[label - 1].lb;
				}

				curRun->lb = label;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::SetFinalLabels(void)
	{
		TLabel *labels = reinterpret_cast<TLabel*>(labels_->data);
		TRun *runs = runs_.data();
		uint runWidth = width_ >> 1;

		#pragma omp parallel for
		for (int run = 0; run < runs_.size(); ++run)
		{
			uint row = run / runWidth * width_;
			TRun curRun = runs[run];

			if (curRun.lb) {
				for (uint i = row + curRun.l; i < row + curRun.r + 1; ++i)
				{
					labels[i] = curRun.lb;
				}
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	// TOCLBinLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	TOCLBinLabeling::TOCLBinLabeling(bool runOnGPU) 
		: binKernel(NULL) 
	{
		cl_device_type devType;
		if (runOnGPU)
			devType = CL_DEVICE_TYPE_GPU;
		else
			devType = CL_DEVICE_TYPE_CPU;

		Init(devType, "", "LabelingAlgs.cl");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLBinLabeling::InitKernels(void) 
	{
		cl_int clError;

		binKernel = clCreateKernel(State.program, "BinLabelingKernel", &clError);

		THROW_IF_OCL(clError, "TOCLBinLabeling::InitKernels");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLBinLabeling::FreeKernels(void) 
	{
		if (binKernel)
			clReleaseKernel(binKernel);
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLBinLabeling::DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth,
		unsigned int imgHeight, TCoherence Coherence)
	{
		cl_int clError;
		size_t workSize = pixels.Buffer().size();

		clError = clSetKernelArg(binKernel, 0, sizeof(cl_mem), (void*)&pixels.buffer);
		clError |= clSetKernelArg(binKernel, 1, sizeof(cl_mem), (void*)&labels.buffer);

		THROW_IF_OCL(clError, "TOCLBinLabeling::DoOCLLabel");

		clEnqueueNDRangeKernel(State.queue, binKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
	}

	///////////////////////////////////////////////////////////////////////////////
	// TOCLLabelDistribution declaration
	///////////////////////////////////////////////////////////////////////////////

	TOCLLabelDistribution::TOCLLabelDistribution(bool runOnGPU)
		: initKernel(NULL),
		  scanKernel(NULL),
		  analizeKernel(NULL)
	{
		cl_device_type devType;
		if (runOnGPU)
			devType = CL_DEVICE_TYPE_GPU;
		else
			devType = CL_DEVICE_TYPE_CPU;

		Init(devType, "", "LabelingAlgs.cl");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelDistribution::InitKernels(void) 
	{
		cl_int clError;

		initKernel = clCreateKernel(State.program, "DistrInitKernel", &clError);
		scanKernel = clCreateKernel(State.program, "DistrScanKernel", &clError);
		analizeKernel = clCreateKernel(State.program, "DistrAnalizeKernel", &clError);

		THROW_IF_OCL(clError, "TOCLLabelDistribution::InitKernels");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelDistribution::FreeKernels(void) 
	{
		if (initKernel)		clReleaseKernel(initKernel);
		if (scanKernel)		clReleaseKernel(scanKernel);
		if (analizeKernel)	clReleaseKernel(analizeKernel);
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelDistribution::DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth,
		unsigned int imgHeight, TCoherence Coherence)
	{
		cl_int clError;
		size_t const workSize = imgWidth * imgHeight;

		// Initialization
		clError = clSetKernelArg(initKernel, 0, sizeof(cl_mem), (void*)&pixels.buffer);
		clError |= clSetKernelArg(initKernel, 1, sizeof(cl_mem), (void*)&labels.buffer);
		THROW_IF_OCL(clError, "TOCLLabelDistribution::DoOCLLabel");

		clError = clEnqueueNDRangeKernel(State.queue, initKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
		THROW_IF_OCL(clError, "TOCLLabelDistribution::DoOCLLabel");

		// Labeling
		TOCLBuffer<char> noChanges(*this, WRITE_ONLY, 1);

		clError = clSetKernelArg(scanKernel, 0, sizeof(cl_mem), (void*)&labels.buffer);
		clError |= clSetKernelArg(scanKernel, 1, sizeof(unsigned int), (void*)&imgWidth);
		clError |= clSetKernelArg(scanKernel, 2, sizeof(unsigned int), (void*)&imgHeight);
		clError |= clSetKernelArg(scanKernel, 3, sizeof(cl_mem), (void*)&noChanges.buffer);
		clError |= clSetKernelArg(analizeKernel, 0, sizeof(cl_mem), (void*)&labels.buffer);
		THROW_IF_OCL(clError, "TOCLLabelDistribution::DoOCLLabel");

		unsigned int iter = 0;
		do
		{
			noChanges[0] = 1;
			noChanges.Push();

			clError |= clEnqueueNDRangeKernel(State.queue, scanKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
			clError |= clEnqueueNDRangeKernel(State.queue, analizeKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);

			noChanges.Pull();
		} while (!noChanges[0]);
	}

	///////////////////////////////////////////////////////////////////////////////
	// TOCLRunEquivLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	TOCLRunEquivLabeling::TOCLRunEquivLabeling(bool runOnGPU)
		: findRunsKernel(NULL),
		findNeibKernel(NULL),
		scanKernel(NULL),
		analizeKernel(NULL),
		labelKernel(NULL)
	{
		cl_device_type devType;
		if (runOnGPU)
			devType = CL_DEVICE_TYPE_GPU;
		else
			devType = CL_DEVICE_TYPE_CPU;

		Init(devType, "", "LabelingAlgs.cl");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::InitKernels(void) 
	{
		cl_int clError;

		initKernel = clCreateKernel(State.program, "REInitRunsKernel", &clError);
		findRunsKernel = clCreateKernel(State.program, "REFindRunsKernel", &clError);
		findNeibKernel = clCreateKernel(State.program, "REFindNeibKernel", &clError);
		scanKernel = clCreateKernel(State.program, "REScanKernel", &clError);
		analizeKernel = clCreateKernel(State.program, "REAnalizeKernel", &clError);
		labelKernel = clCreateKernel(State.program, "RELabelKernel", &clError);

		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::InitKernels");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::FreeKernels(void) {
		if (initKernel)	   clReleaseKernel(initKernel);
		if (findRunsKernel) clReleaseKernel(findRunsKernel);
		if (findNeibKernel) clReleaseKernel(findNeibKernel);
		if (scanKernel)	   clReleaseKernel(scanKernel);
		if (analizeKernel)  clReleaseKernel(analizeKernel);
		if (labelKernel)	   clReleaseKernel(labelKernel);		
	};

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::InitRuns(void)
	{
		cl_int clError;

		// Find runs
		clError = clSetKernelArg(initKernel, 0, sizeof(cl_mem), (void*)&runs);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::InitRuns");

		size_t workSize = height * (width >> 1);
		clError = clEnqueueNDRangeKernel(State.queue, initKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::InitRuns");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::FindRuns(void)
	{
		cl_int clError;

		// Find runs
		clError  = clSetKernelArg(findRunsKernel, 0, sizeof(cl_mem), (void*)&pix->buffer);
		clError |= clSetKernelArg(findRunsKernel, 1, sizeof(cl_mem), (void*)&runs);
		clError |= clSetKernelArg(findRunsKernel, 2, sizeof(unsigned int), (void*)&width);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::FindRuns");

		size_t workSize = height;
		clError = clEnqueueNDRangeKernel(State.queue, findRunsKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::FindRuns");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::FindNeibRuns(void)
	{
		cl_int clError;

		// Find neibour runs
		clError  = clSetKernelArg(findNeibKernel, 0, sizeof(cl_mem), (void*)&runs);
		clError |= clSetKernelArg(findNeibKernel, 1, sizeof(unsigned int), (void*)&width);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::FindNeibRuns");

		size_t workSize = height;
		clError = clEnqueueNDRangeKernel(State.queue, findNeibKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::FindNeibRuns");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::Scan(void)
	{
		cl_int clError;

		// Labeling
		TOCLBuffer<char> noChanges(*this, WRITE_ONLY, 1);

		clError  = clSetKernelArg(scanKernel, 0, sizeof(cl_mem), (void*)&runs);
		clError |= clSetKernelArg(scanKernel, 1, sizeof(unsigned int), (void*)&width);
		clError |= clSetKernelArg(scanKernel, 2, sizeof(cl_mem), (void*)&noChanges.buffer);
		clError |= clSetKernelArg(analizeKernel, 0, sizeof(cl_mem), (void*)&runs);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::Scan");

		size_t workSize = height * (width >> 1);
		do{
			noChanges[0] = 1;
			noChanges.Push();

			clError = clEnqueueNDRangeKernel(State.queue, scanKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
			clError |= clEnqueueNDRangeKernel(State.queue, analizeKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);

			noChanges.Pull();
		} while (!noChanges[0]);

		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::Scan");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::SetFinalLabels(void)
	{
		cl_int clError;

		// Find neibour runs
		clError  = clSetKernelArg(labelKernel, 0, sizeof(cl_mem), (void*)&runs);
		clError |= clSetKernelArg(labelKernel, 1, sizeof(cl_mem), (void*)&lb->buffer);
		clError |= clSetKernelArg(labelKernel, 2, sizeof(unsigned int), (void*)&width);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::SetFinalLabels");

		size_t workSize = height * (width >> 1);
		clError = clEnqueueNDRangeKernel(State.queue, labelKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::SetFinalLabels");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth,
		unsigned int imgHeight, TCoherence Coherence)
	{
		cl_int clError;

		this->pix = &pixels;
		this->lb = &labels;
		this->height = imgHeight;
		this->width = imgWidth;

		// Initialization
		runs = clCreateBuffer(State.context, CL_MEM_READ_WRITE,
			sizeof(TRun) * imgHeight * (imgWidth >> 1), NULL, &clError);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::DoOCLLabel");

		InitRuns();
		FindRuns();
		FindNeibRuns();
		Scan();
		SetFinalLabels();

		clReleaseMemObject(runs);
	}

	///////////////////////////////////////////////////////////////////////////////

} /* LabelingTools */

//#pragma optimize("", on);