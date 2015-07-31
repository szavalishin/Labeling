//Labeling Alorithms
//Copyright (c) by Sergey Zavalishin 2010-2015
//
//Contains algorithms for image labeling.

#ifndef LABELING_ALGS_HPP_
#define LABELING_ALGS_HPP_

#include "LabelingTools.hpp"
#include <memory>
#include <vector>

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
	// TOpenCVLabeing :: OpenCV 3.x.x labeling algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TOpenCVLabeling final : public ILabeling
	{
	private:
		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TBlockGranaLabeing :: Grana's block labeling
	///////////////////////////////////////////////////////////////////////////////

	class TBlockGranaLabeling final : public ILabeling
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

	class TRunLabeling final : public ILabeling
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

	class TLabelDistribution final : public ILabeling
	{
	private:
		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;

		virtual void InitMap(const TImage& pixels, TImage& labels);
		virtual bool Scan(TImage& labels, TCoherence coh);
		virtual void Analyze(TImage& labels);

		inline TLabel MinLabel(TLabel lb1, TLabel lb2) const;
		inline TLabel MinNWSELabel(const TLabel* labels, uint pos, uint width, uint maxPos, TCoherence coh) const;
		inline TLabel GetLabel(const TLabel* labels, uint pos, uint maxPos) const;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TLabelEquivalenceX2 :: OpenMP Label Equivalence X2 algorithm
	///////////////////////////////////////////////////////////////////////////////

	typedef struct
	{
		TLabel lb;		// Super pixel label
		char conn;		// Super pixel neighbor connectivity:
						// 1 2 3
						// 0 x 4
						// 7 6 5
	} TSPixel;

	class TLabelEquivalenceX2 final : public ILabeling
	{	
	private:				
		struct TSPixels {
			std::vector<TSPixel> data;
			size_t w, h;

			TSPixels(size_t width, size_t height) : w(width), h(height), data(width * height) { /* Empty */ }
			inline TSPixel& operator[](size_t pos) { return const_cast<TSPixel&>(static_cast<const TSPixels&>(*this).operator[](pos)); }
			inline const TSPixel& operator[](size_t pos) const { return data[pos]; }
		};

		virtual TSPixels InitSPixels(const TImage& pixels);
		virtual bool Scan(TSPixels& sPixels);
		virtual void Analyze(TSPixels& sPixels);
		virtual void SetFinalLabels(const TImage& pixels, TImage& labels, const TSPixels& sPixels);

		inline TLabel MinSPixLabel(const TSPixels& sPixels, int x, int y);
		inline TLabel GetBlockLabel(const TSPixel *sPix, bool conn, int px, int py, int xshift, int yshift, int w, int h);

		void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TRunEqivLabeling :: OpenMP Run Equivalence algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TRunEqivLabeling final : public ILabeling
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

		TRun *runs_;
		uint *runNum_;
		uint width_, height_, size_;
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

	class TOCLLabelDistribution final : public IOCLLabeling
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
	// TOCLLabelEquivalenceX2 :: OCL Label Equivalence X2 algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TOCLLabelEquivalenceX2 final : public IOCLLabeling
	{
	public:
		TOCLLabelEquivalenceX2(bool runOnGPU = true);

	private:
		cl_kernel initKernel,
				  scanKernel,
				  analyzeKernel,
				  setFinalLabelsKernel;				

		TOCLBuffer<TPixel> *pix;
		TOCLBuffer<TLabel> *lb;

		cl_mem sLabels, sConn;

		unsigned int imgWidth, imgHeight, spWidth, spHeight;

		virtual void InitKernels(void) override;
		virtual void FreeKernels(void) override;

		void InitSPixels(void);
		void LabelSPixels(void);
		void SetFinalLabels(void);
		void FreeSPixels(void);

		void DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth,
			unsigned int imgHeight, TCoherence Coherence) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TOCLRunEquivLabeling :: OCL Run Equivalence algorithm
	///////////////////////////////////////////////////////////////////////////////

	class TOCLRunEquivLabeling final : public IOCLLabeling
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

		cl_mem runs, runNum;

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

	///////////////////////////////////////////////////////////////////////////////
	// TOCLBinLabeling3D :: OCL Binarization for 3D images
	///////////////////////////////////////////////////////////////////////////////

	class TOCLBinLabeling3D final : public IOCLLabeling3D
	{
	public:
		TOCLBinLabeling3D(bool runOnGPU = true);

	private:
		cl_kernel binKernel;

		void DoOCLLabel3D(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, uint imgWidth, uint imgHeight, uint imgDepth) override;

		virtual void InitKernels(void) override;
		virtual void FreeKernels(void) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TOCLLabelEquivalence3D :: OCL Label Equivalence algorithm for 3D images
	///////////////////////////////////////////////////////////////////////////////

	class TOCLLabelEquivalence3D final : public IOCLLabeling3D
	{
	public:
		TOCLLabelEquivalence3D(bool runOnGPU = true);

	private:
		cl_kernel initKernel,
				  scanKernel,
				  analyzeKernel;

		virtual void InitKernels(void) override;
		virtual void FreeKernels(void) override;

		void DoOCLLabel3D(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, 
			uint imgWidth, uint imgHeight, uint imgDepth) override;
	};

	///////////////////////////////////////////////////////////////////////////////
	// TOCLBlockEquivalence3D :: OCL Block Equivalence for 3D images
	///////////////////////////////////////////////////////////////////////////////

	class TOCLBlockEquivalence3D final : public IOCLLabeling3D
	{
	public:
		TOCLBlockEquivalence3D(bool runOnGPU = true);

	private:
		cl_kernel initKernel,
			scanKernel,
			analyzeKernel,
			setFinalLabelsKernel;

		TOCLBuffer<TPixel> *pix;
		TOCLBuffer<TLabel> *lb;

		cl_mem sLabels, sConn;

		unsigned int 
			imgWidth, imgHeight, imgDepth, 
			spWidth, spHeight, spDepth;

		virtual void InitKernels(void) override;
		virtual void FreeKernels(void) override;

		void InitSPixels(void);
		void LabelSPixels(void);
		void SetFinalLabels(void);
		void FreeSPixels(void);

		void DoOCLLabel3D(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, 
			uint imgWidth, uint imgHeight, uint imgDepth) override;
	};

} /* LabelingTools */

#endif /* LABELING_ALGS_HPP_ */