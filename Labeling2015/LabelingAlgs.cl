#ifndef LABELING_ALGS_CL_
#define LABELING_ALGS_CL_

#include "C:/!Proj/GPU Labeling/Code/Labeling2015/Labeling2015/Labeling2015/OCLCommons.cl"

//-------------------------------------------------------------------------
// TOCLBinLabeling kernels
//-------------------------------------------------------------------------

__kernel void BinLabelingKernel(
	__global TPixel	*pixels, // Image pixels
	__global TLabel	*labels	 // Image labels
	)
{
	uint id = get_global_id(0);
	labels[id] = pixels[id] > 0 ? 1 : 0;
}
//-------------------------------------------------------------------------
// TOCLLabelDistribution kernels
//-------------------------------------------------------------------------

__kernel void DistrInitKernel(
	__global TPixel	*pixels,	// Intermediate buffers
	__global TLabel	*labels		// Image labels
	)
{
	const size_t pos = get_global_id(0);
	labels[pos] = pixels[pos] ? pos : 0;
}

__kernel void DistrScanKernel(
	__global TLabel	*labels,	// Image labels
			 uint	width,		// Image width
			 uint	height,		// Image height
	__global char	*noChanges	// Shows if no pixels were changed
	)
{
	const size_t pos = get_global_id(0);
	
	const size_t size = width * height;
	TLabel label = labels[pos];

	if(label)
	{
		TLabel minLabel = MinNWSELabel(labels, pos, width, size);
		
		if(minLabel < label)
		{
			TLabel tmpLabel = labels[label];
			labels[label] = min(tmpLabel, minLabel);
			*noChanges = 0;
		}
	}
}

__kernel void DistrAnalizeKernel(__global TLabel *labels)
{
	const size_t pos = get_global_id(0);

	TLabel label = labels[pos];
	
	if(label){
		TLabel curLabel = labels[label];
		while(curLabel != label)
		{
			label = labels[curLabel];
			curLabel = labels[label];
		}

		labels[pos] = label;
	}
}

//-------------------------------------------------------------------------
// TOCLRunEquivLabeling kernels
//-------------------------------------------------------------------------

	// TRun structure:
	//                   tl          tr 
	//                   v           v
	// Top row:      000 0000 000000 000000000 
	// Current row:   l -> 00000000000000 <- r
	// Bottom row:    0000000000       0000000000
	//                ^                ^
	//                bl               br  
typedef struct
{
	uint l, r;		// Left and right run positions
} TRunSize;

typedef struct
{
	TLabel lb;		// Run label
	uint l, r;		// Run l and r
	TRunSize top;	// Top row tl and tr
	TRunSize bot;	// Bottom row bl and br
} TRun;

__kernel void REInitRunsKernel(
	__global TRun	*runs	// Image runs
	)
{
	const size_t pos = get_global_id(0);
	
	runs[pos].lb	=
	runs[pos].l		=
	runs[pos].r		= 0;
	runs[pos].top.l = 1; // l > r means that there is no neighbors
	runs[pos].top.r = 0;
	runs[pos].bot.l = 1;
	runs[pos].bot.r = 0;
}

__kernel void REFindRunsKernel(
	__global TPixel	*pixels,	// Image pixels
	__global TRun	*runs,		// Image runs
			 uint	width		// Image width
	)
{
	const size_t row = get_global_id(0);
	
	uint pixPos = row * width;
	uint rowPos = row * (width >> 1);

	__global TPixel *curPix = pixels + pixPos;
	__global TRun *curRun = runs + rowPos;

	for(uint pos = 0, runPos = 0; pos < width; ++pos) 
	{
		if(*curPix) {
			if(!curRun->lb) {
				curRun->lb = rowPos + ++runPos;
				curRun->l = pos;
			}			
			if(pos == width - 1) {
				curRun->r = pos;
			}
		} else {
			if(curRun->lb) {
				curRun->r = pos - 1;
				++curRun;
			}
		}
		++curPix;
	}
}

int IsNeib(__global const TRun *r1, __global const TRun *r2)
{
	return 
		(r1->l >= r2->l && r1->l <= r2->r) ||
		(r1->r >= r2->l && r1->r <= r2->r) ||
		(r2->r >= r1->l && r2->r <= r1->r) ||
		(r2->r >= r1->l && r2->r <= r1->r);
}

void FindNeibRuns(__global TRun *curRun, 
				  __global TRunSize *neibSize, 
				  __global const TRun *neibRow, 
				  uint *neibPos, uint runWidth)
{
	int noNeib = 0;

	while(*neibPos < runWidth && !noNeib)
	{
		if(IsNeib(curRun, neibRow + *neibPos)) {
			if(neibSize->l > neibSize->r) {
				neibSize->l = neibRow[*neibPos].lb - 1;
			}
			neibSize->r = neibRow[*neibPos].lb - 1;

			if(*neibPos + 1 < runWidth   && 
				neibRow[*neibPos + 1].lb &&
				neibRow[*neibPos + 1].l <= curRun->r) 
			{
				++(*neibPos);	
			} else {
				noNeib = 1;
			}
		} else {
			if(neibRow[*neibPos].r < curRun->l) {
				++(*neibPos);
			} else {
				noNeib = 1;
			}
		}
	}
}

void FindNeib(__global TRun *curRun, 
			  __global TRunSize *neibSize, 
			  __global const TRun *neibRow,
			  uint runWidth)
{
	for(uint i = 0; i < runWidth && neibRow[i].lb; ++i)
	{
		if(IsNeib(curRun, neibRow + i)) {
			if(neibSize->l > neibSize->r) {
				neibSize->l = neibRow[i].lb - 1;
			}
			neibSize->r = neibRow[i].lb - 1;
		}
	}
}

__kernel void REFindNeibKernel(
	__global TRun	*runs,	// Intermediate buffers
			 uint	width	// Image width
	)
{
	const size_t row = get_global_id(0);
	const size_t height = get_global_size(0);

	uint runWidth = width >> 1;

	__global TRun *curRun = runs + row * runWidth;
	const __global TRun *topRow = curRun - runWidth;
	const __global TRun *botRow = curRun + runWidth;

	uint topPos = 0;
	uint botPos = 0;

	for(uint pos = 0; curRun->lb != 0 && pos < runWidth; ++pos)
	{
		if(row > 0) {
			//FindNeib(curRun, &curRun->top, topRow, runWidth);
			FindNeibRuns(curRun, &curRun->top, topRow, &topPos, runWidth);
		}
		
		if(row < height - 1) {
			//FindNeib(curRun, &curRun->bot, botRow, runWidth);
			FindNeibRuns(curRun, &curRun->bot, botRow, &botPos, runWidth);
		}

		++curRun;
	}
}

TLabel MinRunLabel(const __global TRun *runs, uint pos)
{
	TLabel minLabel = UINT_MAX;

	if(runs[pos].top.l <= runs[pos].top.r) {
		for(uint i = runs[pos].top.l; i < runs[pos].top.r + 1; ++i) {
			minLabel = min(minLabel, runs[i].lb);
		}
	}

	if(runs[pos].bot.l <= runs[pos].bot.r) {
		for(uint i = runs[pos].bot.l; i < runs[pos].bot.r + 1; ++i) {
			minLabel = min(minLabel, runs[i].lb);
		}
	}

	return minLabel;
}

__kernel void REScanKernel(
	__global TRun	*runs,		// Image runs
			 uint	width,		// Image width
	__global char	*noChanges	// Shows if no pixels were changed
	)
{
	const size_t pos = get_global_id(0);
	
	uint runWidth = width >> 1;
	TLabel label = runs[pos].lb;

	if(label)
	{
		TLabel minLabel = MinRunLabel(runs, pos);
		
		if(minLabel < label)
		{
			TLabel tmpLabel = runs[label - 1].lb;
			runs[label - 1].lb = min(tmpLabel, minLabel);
			*noChanges = 0;
		}
	}
}

__kernel void REAnalizeKernel(__global TRun *runs)
{
	const size_t pos = get_global_id(0);

	TLabel label = runs[pos].lb;
	
	if(label){
		TLabel curLabel = runs[label - 1].lb;
		while(curLabel != label)
		{
			label = runs[curLabel - 1].lb;
			curLabel = runs[label - 1].lb;
		}

		runs[pos].lb = label;
	}
}

__kernel void RELabelKernel(
	__global TRun	*runs,		// Image runs
	__global TLabel	*labels,	// Image labels
			 uint	width		// Image width
	)
{
	const size_t pos = get_global_id(0);

	uint runWidth = width >> 1;
	uint row = pos / runWidth * width;
	
	if(runs[pos].lb) {
		for(uint i = row + runs[pos].l; i < row + runs[pos].r + 1; ++i)
		{
			labels[i] = runs[pos].lb;
		}
	}
}

#endif /* LABELING_ALGS_CL_ */