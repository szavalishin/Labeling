#ifndef OCL_COMMONS_CL_
#define OCL_COMMONS_CL_

typedef uchar TPixel;
typedef uint  TLabel;

typedef enum TCoherence
{
	COH_4,
	COH_8,
	COH_DEFAULT
} TCoherence;

TPixel GetPixel(__global TPixel *pix, size_t pos, uint maxSize) {
	return pos && pos < maxSize ? pix[pos] : 0;
}

TLabel GetLabel(__global TLabel *labels, size_t pos, uint maxSize) {
	return pos && pos < maxSize ? labels[pos] : 0;
}

TLabel MinLabel(TLabel lb1, TLabel lb2)
{
	if(lb1 && lb2) 
		return min(lb1, lb2);
	
	TLabel lb = max(lb1, lb2);
	
	return lb ? lb : UINT_MAX;
}

TLabel MinNWSELabel(__global TLabel *lb, size_t lbPos, uint width, uint maxPos, TCoherence coh)
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

#endif /* OCL_COMMONS_CL_ */