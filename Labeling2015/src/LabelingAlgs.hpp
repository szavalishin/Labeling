//Labeling Alorithms
//Copyright (c) by Sergey Zavalishin 2010-2015
//
//Contains algorithms for image labeling.

#ifndef LABELING_ALGS_HPP_
#define LABELING_ALGS_HPP_

#include "LabelingTools.hpp"

///////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace stdext;

///////////////////////////////////////////////////////////////////////////////

namespace LabelingTools
{	

	void SetupThreads(char Threads); //sets thread count for next parallel region

	///////////////////////////////////////////////////////////////////////////////
	// TBinLabeing :: Just binarization
	///////////////////////////////////////////////////////////////////////////////
	
	class TBinLabeling final : public ILabeling
	{
	private:
		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TOpenCVLabeing :: OpenCV 3+ labeling algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TOpenCVLabeling : public ILabeling
	{
	private:
		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TBlockGranaLabeing :: Grana's block labeling
	///////////////////////////////////////////////////////////////////////////////

	class TBlockGranaLabeling : public ILabeling
	{
	private:
		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TRunLabeling :: He's run labeling algorithm
	///////////////////////////////////////////////////////////////////////////////
	
	struct TRun
	{
		uint l;			// First pixel in run
		uint r;			// Last pixel in run
		TLabel Label;	// Run's representive label
		uint Row;		// Represents run's row
	};

	typedef vector<TRun*> TRuns; // Represents runs array	

	class TRunLabeling : public ILabeling
	{
	public:
		unsigned int Top;
		unsigned int Bottom;

		// Constructor
		TRunLabeling(void);
		TRunLabeling(unsigned int aTop, unsigned int aBottom);

	private:
		int ConPix; //represents if we need additional 
					//pixels at left and right due to the 8x coherence
		TRun* CurRun; //current run

		TRuns Runs; //runs array
		TRuns LastRow; //represents all runs in upper row
		TRuns CurRow; //represents all runs in current row
		vector<TLabel> Objects; //represents provisional labels as 
								//Objects[Current_Label] = Real_Label
		vector<TLabel> LabelStack;

		//labeling itself
		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;
		//sets run label
		void SetRunLabel(void);
	};

	///////////////////////////////////////////////////////////////////////////////
	// TLabelDistribution :: OpenMP Label Equivalence algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TLabelDistribution : public ILabeling
	{
	private:
		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;

		virtual void InitMap(const TImage& pixels, TImage& labels);
		virtual bool Scan(TImage& labels);
		virtual void Analyze(TImage& labels);

		inline TLabel MinLabel(TLabel lb1, TLabel lb2) const;
		inline TLabel MinNWSELabel(const TLabel* labels, uint pos, uint width, uint maxPos) const;
		inline TLabel GetLabel(const TLabel* labels, uint pos, uint maxPos) const;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TRunEqivLabeling :: OpenMP Run Equivalence algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TRunEqivLabeling : public ILabeling
	{
	private:
		typedef struct
		{
			cl_uint l, r;	// Left and right run positions
		} TRunSize;

		typedef struct
		{
			TLabel lb;		// Run label
			cl_uint l, r;	// Run l and r
			TRunSize top;	// Top row tl and tr
			TRunSize bot;	// Bottom row bl and br
		} TRun;

		vector<TRun> runs_;
		uint width_, height_;
		const TImage *pixels_; 
		TImage *labels_;

		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;

		virtual void InitRuns(void);
		virtual void FindRuns(void);
		virtual void FindNeibRuns(void);
		virtual void Scan(void);
		virtual void SetFinalLabels(void);

		inline bool IsNeib(const TRun *r1, const TRun *r2) const;
		inline void FindNeibRuns(TRun *curRun, TRunSize *neibSize, const TRun *neibRow, uint *neibPos, uint runWidth);
		inline TLabel MinRunLabel(uint pos);
		inline bool ScanRuns(void);
		inline void AnalyzeRuns(void);
	};

	///////////////////////////////////////////////////////////////////////////////
	// TOCLBinLabeling :: OCL Binarization
	///////////////////////////////////////////////////////////////////////////////

	class TOCLBinLabeling final : public IOCLLabeling 
	{
	public:
		TOCLBinLabeling(bool runOnGPU = true);		

	private:
		cl_kernel binKernel;

		void DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth,
			unsigned int imgHeight, TCoherence Coherence) override;

		virtual void InitKernels(void) override;
		virtual void FreeKernels(void) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TOCLLabelDistribution :: OCL Label Equivalence algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TOCLLabelDistribution : public IOCLLabeling 
	{
	public:
		TOCLLabelDistribution(bool runOnGPU = true);

	private:
		cl_kernel initKernel,
			      scanKernel,
			      analizeKernel;

		virtual void InitKernels(void) override;
		virtual void FreeKernels(void) override;	

		void DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth, 
						unsigned int imgHeight, TCoherence Coherence) override;		
	};

	///////////////////////////////////////////////////////////////////////////////
	// TOCLRunEquivLabeling :: OCL Run Equivalence algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TOCLRunEquivLabeling : public IOCLLabeling 
	{
	public:
		TOCLRunEquivLabeling(bool runOnGPU = true);		

	private:
		typedef struct
		{
			cl_uint l, r;	// Left and right run positions
		} TRunSize;

		typedef struct
		{
			TLabel lb;		// Run label
			cl_uint l, r;	// Run l and r
			TRunSize top;	// Top row tl and tr
			TRunSize bot;	// Bottom row bl and br
		} TRun;

		cl_kernel initKernel,
				  findRunsKernel,
				  findNeibKernel,
				  scanKernel,
				  analizeKernel,
				  labelKernel;

		cl_mem runs;

		TOCLBuffer<TPixel> *pix;
		TOCLBuffer<TLabel> *lb;

		unsigned int width;
		unsigned int height;

		void DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth, 
						unsigned int imgHeight, TCoherence Coherence) override;

		virtual void InitKernels(void) override;
		virtual void FreeKernels(void) override;

		void InitRuns(void);
		void FindRuns(void);
		void FindNeibRuns(void);
		void Scan(void);
		void SetFinalLabels(void);
	};

} /* LabelingTools */

#endif /* LABELING_ALGS_HPP_ */