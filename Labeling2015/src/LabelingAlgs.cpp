// Labeling Alhorithms declaration
// Copyright (c) by Sergey Zavalishin 2010-2015
//
// See defenition in LabelingAlgs.hpp

#include "LabelingAlgs.hpp"
#include "cvlabeling_imagelab.h"

#include <limits>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

		TPixel *pix = pixels.data;
		TLabel *lb = reinterpret_cast<TLabel*>(labels.data);

#		pragma omp parallel for
		for (int i = 0; i < pixels.total(); ++i)
		{
			if (pixels.data[i])
				lb[i] = 1;			
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
		THROW_IF(coh == COH_4, "TLabelEquivalenceX2::DoLabel : Method does not support 4x connectivity");

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
		if (coh == COH_DEFAULT) coh = COH_4;

		SetupThreads(threads);
		InitMap(pixels, labels);
		
		while (true) {
			if (Scan(labels, coh)) break;
			Analyze(labels);			
		}
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

	TLabel TLabelDistribution::MinNWSELabel(const TLabel* lb, uint lbPos, uint width, uint maxPos, TCoherence coh) const
	{
		if (coh == COH_8)
			return MinLabel(GetLabel(lb, lbPos - 1, maxPos),
				MinLabel(GetLabel(lb, lbPos - width, maxPos),
				MinLabel(GetLabel(lb, lbPos + width, maxPos),
				MinLabel(GetLabel(lb, lbPos - 1 - width, maxPos),
				MinLabel(GetLabel(lb, lbPos + 1 + width, maxPos),
				MinLabel(GetLabel(lb, lbPos - 1 + width, maxPos),
				MinLabel(GetLabel(lb, lbPos + 1 - width, maxPos),
				GetLabel(lb, lbPos + 1, maxPos))))))));
		else
			return MinLabel(GetLabel(lb, lbPos - 1, maxPos),
				MinLabel(GetLabel(lb, lbPos - width, maxPos),
				MinLabel(GetLabel(lb, lbPos + width, maxPos),				
				GetLabel(lb, lbPos + 1, maxPos))));
	}
	
	///////////////////////////////////////////////////////////////////////////////
	
	TLabel TLabelDistribution::GetLabel(const TLabel* labels, uint pos, uint maxPos) const
	{		
		return pos && pos < maxPos ? labels[pos] : 0;
	}

	///////////////////////////////////////////////////////////////////////////////

	bool TLabelDistribution::Scan(TImage& labels, TCoherence coh)
	{
		bool noChanges = true;

		TLabel *lb = reinterpret_cast<TLabel*>(labels.data);

		#pragma omp parallel for
		for (long int i = 0; i < labels.total(); ++i)
		{
			TLabel label = lb[i];

			if (label)
			{
				TLabel minLabel = MinNWSELabel(lb, i, labels.cols, labels.total(), coh);

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
	// TLabelEquivalenceX2 declaration
	///////////////////////////////////////////////////////////////////////////////

	void TLabelEquivalenceX2::DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		THROW_IF(coh == COH_4, "TLabelEquivalenceX2::DoLabel : Method does not support 4x connectivity");		

		SetupThreads(threads);
		TSPixels sPixels = InitSPixels(pixels);
			
		while (true) {
			if (Scan(sPixels)) break;
			Analyze(sPixels);
		}

		SetFinalLabels(pixels, labels, sPixels);
	}

	///////////////////////////////////////////////////////////////////////////////

	inline bool TestBit(const TPixel *pix, int px, int py, int xshift, int yshift, int w, int h)
	{
		return pix[px + xshift + (py + yshift) * w];
	}

	inline ushort CheckNeibPixABC(bool C1, bool C2) {
		return (C1 ? 3 : 0) | (C2 ? 0x18 : 0) | (C1 && C2) << 2;
	}

	inline ushort CheckNeibPixD(bool C1, bool C2) {
		return (C1 ? 3 : 0) << 9 | (C2 ? 3 : 0) | (C1 && C2) << 11;
	}

	TLabelEquivalenceX2::TSPixels TLabelEquivalenceX2::InitSPixels(const TImage& pixels)
	{
		TSPixels sPixels(ceil(static_cast<float>(pixels.cols) / 2), ceil(static_cast<float>(pixels.rows) / 2));
		int w = pixels.cols, h = pixels.rows;
		TPixel *pix = pixels.data;
		
		#pragma omp parallel for
		for (int spy = 0; spy < sPixels.h; ++spy) {
			for (int spx = 0; spx < sPixels.w; ++spx) {
				size_t spos = spx + spy * sPixels.w;
				size_t px = spx * 2, py = spy * 2;
				size_t ppos = px + py * w;

				TSPixel sPix = {0, 0};

				// 2 3 4 5
				// 1 a b 6
				// 0 d c 7
				// B A 9 8
				ushort testPattern = 0;
				if (pix[ppos])										testPattern  = CheckNeibPixABC(px, py);
				if (px + 1 < w && pix[ppos + 1])					testPattern |= CheckNeibPixABC(py, px + 2 < w) << 3;
				if (px + 1 < w && py + 1 < h && pix[ppos + 1 + w])	testPattern |= CheckNeibPixABC(px + 2 < w, py + 2 < h) << 6;
				if (py + 1 < h && pix[ppos + w])					testPattern |= CheckNeibPixD(py + 2 < h, px);

				if (testPattern) {
					sPix.lb = spos;

					if ((testPattern & 1 << 0 && TestBit(pix, px, py, -1, 1, w, h)) ||
						(testPattern & 1 << 1 && TestBit(pix, px, py, -1, 0, w, h)))
						sPix.conn = 1;
					if ((testPattern & 1 << 2 && TestBit(pix, px, py, -1, -1, w, h)))
						sPix.conn |= 1 << 1;
					if ((testPattern & 1 << 3 && TestBit(pix, px, py, 0, -1, w, h)) ||
						(testPattern & 1 << 4 && TestBit(pix, px, py, 1, -1, w, h)))
						sPix.conn |= 1 << 2;
					if ((testPattern & 1 << 5 && TestBit(pix, px, py, 2, -1, w, h)))
						sPix.conn |= 1 << 3;
					if ((testPattern & 1 << 6 && TestBit(pix, px, py, 2, 0, w, h)) ||
						(testPattern & 1 << 7 && TestBit(pix, px, py, 2, 1, w, h)))
						sPix.conn |= 1 << 4;
					if ((testPattern & 1 << 8 && TestBit(pix, px, py, 2, 2, w, h)))
						sPix.conn |= 1 << 5;
					if ((testPattern & 1 << 9 && TestBit(pix, px, py, 1, 2, w, h)) ||
						(testPattern & 1 << 10 && TestBit(pix, px, py, 0, 2, w, h)))
						sPix.conn |= 1 << 6;
					if ((testPattern & 1 << 11 && TestBit(pix, px, py, -1, 2, w, h)))
						sPix.conn |= 1 << 7;
				}

				sPixels[spos] = sPix;
			}
		}

		return sPixels;
	}

	///////////////////////////////////////////////////////////////////////////////

	bool TLabelEquivalenceX2::Scan(TSPixels& sPixels)
	{
		bool noChanges = true;
		TSPixel *sPix = sPixels.data.data();

		#pragma omp parallel for
		for (int y = 0; y < sPixels.h; ++y) {
			for (int x = 0; x < sPixels.w; ++x) {
				TLabel label = sPix[x + y * sPixels.w].lb;

				if (label) {
					TLabel minLabel = MinSPixLabel(sPixels, x, y);

					if (minLabel < label) {
						TLabel tmpLabel = sPix[label].lb;
						sPix[label].lb = min(tmpLabel, minLabel);
						noChanges = false;
					}
				}
			}
		}

		return noChanges;
	}

	///////////////////////////////////////////////////////////////////////////////

	inline TLabel TLabelEquivalenceX2::GetBlockLabel(const TSPixel *sPix, bool conn, int px, int py, int xshift, int yshift, int w, int h)
	{		
		return conn ? sPix[px + xshift + (py + yshift) * w].lb : UINT_MAX;
	}

	TLabel TLabelEquivalenceX2::MinSPixLabel(const TSPixels& sPixels, int x, int y)
	{		
		TLabel minLabel;
		const TSPixel *sPix = sPixels.data.data();
		uchar conn = sPix[x + y * sPixels.w].conn;
		int w = sPixels.w, h = sPixels.h;

		minLabel = min(GetBlockLabel(sPix, conn & (1 << 0), x, y, -1, 0, w, h),
			min(GetBlockLabel(sPix, conn & (1 << 1), x, y, -1, -1, w, h),
			min(GetBlockLabel(sPix, conn & (1 << 2), x, y, 0, -1, w, h),
			min(GetBlockLabel(sPix, conn & (1 << 3), x, y, 1, -1, w, h),
			min(GetBlockLabel(sPix, conn & (1 << 4), x, y, 1, 0, w, h),
			min(GetBlockLabel(sPix, conn & (1 << 5), x, y, 1, 1, w, h),
			min(GetBlockLabel(sPix, conn & (1 << 6), x, y, 0, 1, w, h),
			GetBlockLabel(sPix, conn & (1 << 7), x, y, -1, 1, w, h))))))));		

		return minLabel;
	}

	///////////////////////////////////////////////////////////////////////////////

	void TLabelEquivalenceX2::Analyze(TSPixels& sPixels)
	{		
		#pragma omp parallel for
		for (long sPos = 0; sPos < sPixels.w * sPixels.h; ++sPos) {
			TLabel label = sPixels[sPos].lb;

			if (label) {
				TLabel curLabel = sPixels[label].lb;
				while (curLabel != label) {
					label = sPixels[curLabel].lb;
					curLabel = sPixels[label].lb;
				}

				sPixels[sPos].lb = label;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	void TLabelEquivalenceX2::SetFinalLabels(const TImage& pixels, TImage& labels, const TSPixels& sPixels)
	{
		//labels = TImage(pixels.rows, pixels.cols, CV_32SC1, cv::Scalar(0));
		TLabel *lb = reinterpret_cast<TLabel*>(labels.data);
		TPixel *pix = pixels.data;

		#pragma omp parallel for
		for (int y = 0; y < pixels.rows; ++y) {
			for (int x = 0; x < pixels.cols; ++x) {				
				const size_t sPos = x / 2 + y / 2 * sPixels.w;
				const size_t pos = x + y * pixels.cols;

				TLabel label = sPixels[sPos].lb;

				if (pix[pos]) {
					lb[pos] = label;
				}
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	// TRunEqivLabeling declaration
	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh)
	{
		THROW_IF(coh == COH_4, "TRunEqivLabeling::DoLabel : 4x connectivity is not implemented for this method");

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

		delete runs_;
		delete runNum_;
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::InitRuns(void)
	{
		size_ = height_ * (width_ >> 1);

		runs_ = new TRun[size_];
		runNum_ = new uint[height_];

		memset(runNum_, 0, sizeof(uint) * height_);
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::FindRuns(void)
	{		
#		pragma omp parallel for schedule(dynamic)
		for (int row = 0; row < height_; ++row)
		{
			uint pixPos = row * width_;
			uint rowPos = row * (width_ >> 1);

			TPixel *curPix = reinterpret_cast<TPixel*>(pixels_->data) + pixPos;
			TRun *curRun = runs_ + row * (width_ >> 1);

			uint runPos = 0;
			curRun->lb = 0;
			for (uint pos = 0; pos < width_; ++pos)
			{
				if (*curPix) {
					if (!curRun->lb) {						
						curRun->lb = rowPos + ++runPos;
						curRun->l = pos;
					}
					if (pos == width_ - 1) {
						curRun->r = pos;						
						++curRun;
						curRun->lb = 0;
					}
				} else {
					if (curRun->lb) {
						curRun->r = pos - 1;						
						++curRun;
						curRun->lb = 0;
					}
				}
				++curPix;				
			}
			runNum_[row] = runPos;			
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
		neibSize->l = 1;
		neibSize->r = 0;

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
		TRun *runs = runs_;
		
#		pragma omp parallel for
		for (int row = 0; row < height_; ++row)
		{
			TRun *curRun = runs + row * runWidth;
			const TRun *topRow = curRun - runWidth;
			const TRun *botRow = curRun + runWidth;

			uint topPos = 0;
			uint botPos = 0;

			for (uint pos = 0; pos < runNum_[row]; ++pos)
			{
				if (row > 0) {					
					FindNeibRuns(curRun, &curRun->top, topRow, &topPos, runNum_[row - 1]);
				}else{
					curRun->top.l = 1;
					curRun->top.r = 0;
				}

				if (row < height_ - 1) {					
					FindNeibRuns(curRun, &curRun->bot, botRow, &botPos, runNum_[row + 1]);
				}else{
					curRun->bot.l = 1;
					curRun->bot.r = 0;
				}

				++curRun;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	TLabel TRunEqivLabeling::MinRunLabel(uint pos)
	{
		TLabel minLabel = UINT_MAX;
		TRun *runs = runs_;
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
		while (true) {
			if (ScanRuns()) break;
			AnalyzeRuns();
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	bool TRunEqivLabeling::ScanRuns(void)
	{
		bool noChanges = true;
		TRun *runs = runs_;
		
#		pragma omp parallel for
		for (int row = 0; row < height_; ++row)
		{
			for (int pos = 0; pos < runNum_[row]; ++pos)
			{				
				TLabel label = runs[row * (width_ >> 1) + pos].lb;

				if (label)
				{
					TLabel minLabel = MinRunLabel(row * (width_ >> 1) + pos);
					
					if (minLabel < label)
					{
						TLabel tmpLabel = runs[label - 1].lb;						

						runs[label - 1].lb = min(tmpLabel, minLabel);
						noChanges = false;
					}
				}
			}
		}

		return noChanges;
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::AnalyzeRuns(void)
	{
		TRun *runs = runs_;

#		pragma omp parallel for
		for (int row = 0; row < height_; ++row)
		{
			for (int pos = 0; pos < runNum_[row]; ++pos)
			{
				TRun *curRun = &runs[row * (width_ >> 1) + pos];
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
	}

	///////////////////////////////////////////////////////////////////////////////

	void TRunEqivLabeling::SetFinalLabels(void)
	{
		TLabel *labels = reinterpret_cast<TLabel*>(labels_->data);
		TRun *runs = runs_;
		uint runWidth = width_ >> 1;

#		pragma omp parallel for
		for (int row = 0; row < height_; ++row)
		{
			for (int run = 0; run < runNum_[row]; ++run)
			{				
				TRun curRun = runs[row * (width_ >> 1) + run];

				if (curRun.lb) {
					for (uint i = curRun.l; i < curRun.r + 1; ++i)
					{
						labels[row * width_ + i] = curRun.lb;
					}
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
		unsigned int imgHeight, TCoherence coh)
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
		clError |= clSetKernelArg(scanKernel, 3, sizeof(TCoherence), (void*)&coh);
		clError |= clSetKernelArg(scanKernel, 4, sizeof(cl_mem), (void*)&noChanges.buffer);	
		clError |= clSetKernelArg(analizeKernel, 0, sizeof(cl_mem), (void*)&labels.buffer);
		THROW_IF_OCL(clError, "TOCLLabelDistribution::DoOCLLabel");

		unsigned int iter = 0;
		while (true) {
			noChanges[0] = 1;
			noChanges.Push();

			clError |= clEnqueueNDRangeKernel(State.queue, scanKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);

			noChanges.Pull();
			if (noChanges[0]) break;

			clError |= clEnqueueNDRangeKernel(State.queue, analizeKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	// TOCLLabelEquivalenceX2 declaration
	///////////////////////////////////////////////////////////////////////////////

	TOCLLabelEquivalenceX2::TOCLLabelEquivalenceX2(bool runOnGPU)
		: initKernel(NULL),
		  scanKernel(NULL),
		  analyzeKernel(NULL),
		  setFinalLabelsKernel(NULL)
	{
		cl_device_type devType;
		if (runOnGPU)
			devType = CL_DEVICE_TYPE_GPU;
		else
			devType = CL_DEVICE_TYPE_CPU;

		Init(devType, "", "LabelingAlgs.cl");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelEquivalenceX2::InitKernels(void)
	{
		cl_int clError;

		initKernel = clCreateKernel(State.program, "LBEQ2_Init", &clError);
		scanKernel = clCreateKernel(State.program, "LBEQ2_Scan", &clError);
		analyzeKernel = clCreateKernel(State.program, "LBEQ2_Analyze", &clError);
		setFinalLabelsKernel = clCreateKernel(State.program, "LBEQ2_SetFinalLabels", &clError);

		THROW_IF_OCL(clError, "TOCLLabelEquivalenceX2::InitKernels");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelEquivalenceX2::FreeKernels(void)
	{
		if (initKernel)				clReleaseKernel(initKernel);
		if (scanKernel)				clReleaseKernel(scanKernel);
		if (analyzeKernel)			clReleaseKernel(analyzeKernel);
		if (setFinalLabelsKernel)	clReleaseKernel(setFinalLabelsKernel);
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelEquivalenceX2::InitSPixels(void)
	{
		cl_int clError;

		spWidth = ceil(static_cast<float>(imgWidth) / 2);
		spHeight = ceil(static_cast<float>(imgHeight) / 2);		

		sLabels = clCreateBuffer(State.context, CL_MEM_READ_WRITE, sizeof(TLabel) * spHeight * spWidth, NULL, &clError);
		sConn   = clCreateBuffer(State.context, CL_MEM_READ_WRITE, sizeof(char) * spHeight * spWidth, NULL, &clError);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::DoOCLLabel");

		clError  = clSetKernelArg(initKernel, 0, sizeof(cl_mem), (void*)&pix->buffer);				
		clError |= clSetKernelArg(initKernel, 1, sizeof(cl_mem), (void*)&sLabels);
		clError |= clSetKernelArg(initKernel, 2, sizeof(cl_mem), (void*)&sConn);
		clError |= clSetKernelArg(initKernel, 3, sizeof(cl_mem), (void*)&imgWidth);		
		clError |= clSetKernelArg(initKernel, 4, sizeof(cl_mem), (void*)&imgHeight);
		THROW_IF_OCL(clError, "TOCLLabelEquivalenceX2::InitSPixels");

		size_t workSize[] = { spWidth, spHeight };
		clError = clEnqueueNDRangeKernel(State.queue, initKernel, 2, NULL, workSize, NULL, 0, NULL, NULL);
		THROW_IF_OCL(clError, "TOCLLabelEquivalenceX2::InitSPixels");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelEquivalenceX2::LabelSPixels(void)
	{
		cl_int clError;
		const size_t scanWorkSize[] = { spWidth, spHeight };
		const size_t analyzeWorkSize[] = { scanWorkSize[0] * scanWorkSize[1] };
	
		TOCLBuffer<char> noChanges(*this, WRITE_ONLY, 1);		

		clError  = clSetKernelArg(scanKernel, 0, sizeof(cl_mem), (void*)&sLabels);
		clError |= clSetKernelArg(scanKernel, 1, sizeof(cl_mem), (void*)&sConn);
		clError |= clSetKernelArg(scanKernel, 2, sizeof(cl_mem), (void*)&noChanges.buffer);
		clError |= clSetKernelArg(analyzeKernel, 0, sizeof(cl_mem), (void*)&sLabels);
		THROW_IF_OCL(clError, "TOCLLabelEquivalenceX2::LabelSPixels");
		
		while (true) {
			noChanges[0] = 1;
			noChanges.Push();

			clError |= clEnqueueNDRangeKernel(State.queue, scanKernel, 2, NULL, scanWorkSize, NULL, 0, NULL, NULL);

			noChanges.Pull();
			if (noChanges[0]) break;

			clError |= clEnqueueNDRangeKernel(State.queue, analyzeKernel, 1, NULL, analyzeWorkSize, NULL, 0, NULL, NULL);					
		}
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelEquivalenceX2::SetFinalLabels(void)
	{
		cl_int clError;
		size_t workSize[] = { imgWidth, imgHeight };

		clError = clSetKernelArg(setFinalLabelsKernel, 0, sizeof(cl_mem), (void*)&pix->buffer);
		clError |= clSetKernelArg(setFinalLabelsKernel, 1, sizeof(cl_mem), (void*)&lb->buffer);
		clError |= clSetKernelArg(setFinalLabelsKernel, 2, sizeof(cl_mem), (void*)&sLabels);
		THROW_IF_OCL(clError, "TOCLLabelEquivalenceX2::SetFinalLabels");

		clError |= clEnqueueNDRangeKernel(State.queue, setFinalLabelsKernel, 2, NULL, workSize, NULL, 0, NULL, NULL);
		THROW_IF_OCL(clError, "TOCLLabelEquivalenceX2::SetFinalLabels");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelEquivalenceX2::FreeSPixels(void)
	{
		clReleaseMemObject(sLabels);
		clReleaseMemObject(sConn);
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLLabelEquivalenceX2::DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imWidth,
		unsigned int imHeight, TCoherence coh)
	{
		THROW_IF(coh == COH_4, "TOCLLabelEquivalenceX2::DoOCLLabel : Method does not support 4x connectivity");

		cl_int clError;

		pix = &pixels;
		lb = &labels;

		imgWidth = imWidth;
		imgHeight = imHeight;		

		InitSPixels();
		LabelSPixels();
		SetFinalLabels();
		FreeSPixels();		
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
		clError = clSetKernelArg(initKernel, 0, sizeof(cl_mem), (void*)&runNum);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::InitRuns");

		size_t workSize = height;
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
		clError |= clSetKernelArg(findRunsKernel, 2, sizeof(cl_mem), (void*)&runNum);
		clError |= clSetKernelArg(findRunsKernel, 3, sizeof(unsigned int), (void*)&width);
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
		clError |= clSetKernelArg(findNeibKernel, 1, sizeof(cl_mem), (void*)&runNum);
		clError |= clSetKernelArg(findNeibKernel, 2, sizeof(unsigned int), (void*)&width);
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
		clError |= clSetKernelArg(scanKernel, 1, sizeof(cl_mem), (void*)&runNum);
		clError |= clSetKernelArg(scanKernel, 2, sizeof(unsigned int), (void*)&width);
		clError |= clSetKernelArg(scanKernel, 3, sizeof(cl_mem), (void*)&noChanges.buffer);
		clError |= clSetKernelArg(analizeKernel, 0, sizeof(cl_mem), (void*)&runs);
		clError |= clSetKernelArg(analizeKernel, 1, sizeof(cl_mem), (void*)&runNum);
		clError |= clSetKernelArg(analizeKernel, 2, sizeof(unsigned int), (void*)&width);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::Scan");

		size_t workSize = height;
		while (true) {
			noChanges[0] = 1;
			noChanges.Push();

			clError = clEnqueueNDRangeKernel(State.queue, scanKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);	

			noChanges.Pull();
			if (noChanges[0]) break;

			clError |= clEnqueueNDRangeKernel(State.queue, analizeKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
			
			THROW_IF_OCL(clError, "TOCLRunEquivLabeling::Scan");					
		}

		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::Scan");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::SetFinalLabels(void)
	{
		cl_int clError;

		// Find neibour runs
		clError  = clSetKernelArg(labelKernel, 0, sizeof(cl_mem), (void*)&runs);
		clError |= clSetKernelArg(labelKernel, 1, sizeof(cl_mem), (void*)&runNum);
		clError |= clSetKernelArg(labelKernel, 2, sizeof(cl_mem), (void*)&lb->buffer);
		clError |= clSetKernelArg(labelKernel, 3, sizeof(unsigned int), (void*)&width);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::SetFinalLabels");

		size_t workSize = height;// *(width >> 1);
		clError = clEnqueueNDRangeKernel(State.queue, labelKernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::SetFinalLabels");
	}

	///////////////////////////////////////////////////////////////////////////////

	void TOCLRunEquivLabeling::DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth,
		unsigned int imgHeight, TCoherence coh)
	{
		THROW_IF(coh == COH_4, "TOCLRunEquivLabeling::DoLabel : 4x connectivity is not implemented for this method");

		cl_int clError;

		this->pix = &pixels;
		this->lb = &labels;
		this->height = imgHeight;
		this->width = imgWidth;

		// Initialization
		runs = clCreateBuffer(State.context, CL_MEM_READ_WRITE,
			sizeof(TRun) * imgHeight * (imgWidth >> 1), NULL, &clError);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::DoOCLLabel");

		runNum = clCreateBuffer(State.context, CL_MEM_READ_WRITE,
			sizeof(uint) * imgHeight, NULL, &clError);
		THROW_IF_OCL(clError, "TOCLRunEquivLabeling::DoOCLLabel");

		InitRuns();
		FindRuns();
		FindNeibRuns();
		Scan();
		SetFinalLabels();

		clReleaseMemObject(runs);
		clReleaseMemObject(runNum);
	}

	///////////////////////////////////////////////////////////////////////////////

} /* LabelingTools */

//#pragma optimize("", on);