// Labeling2015.cpp : Defines the entry point for the console application.
//

#include <time.h>

#include <iostream>
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <set>
#include <list>

#include "src/LabelingTools.hpp"
#include "src/LabelingAlgs.hpp"

///////////////////////////////////////////////////////////////////////////////

using namespace boost::filesystem;
using namespace LabelingTools;
using namespace std;

///////////////////////////////////////////////////////////////////////////////

struct Options
{
	std::string inPath;
	std::string outPath;

	int numThreads = MAX_THREADS;
	int cycles = 1;

	std::shared_ptr<ILabeling> labelingAlg;

	bool useGPU = false;
	TCoherence coh = COH_DEFAULT;
	bool quickExit = false;
};

///////////////////////////////////////////////////////////////////////////////

struct ImgTime
{			
	void Reset(void) { min = UINT_MAX; c = avg = max = 0; }	
	
	TTime Min(void) const { return min != UINT_MAX ? min : 0; }
	TTime Max(void) const { return max; }
	TTime Avg(void) const { return static_cast<float>(avg) / c; }
	
	void Add(TTime t) {
		if (min > t) min = t;
		if (max < t) max = t;
		avg += t;
		++c;
	}

	ImgTime(void) { Reset(); }
	ImgTime& operator+(const ImgTime& other) {
		this->avg += other.avg;
		this->min = std::min(this->min, other.min);
		this->max = std::max(this->max, other.max);
		this->c += other.c;

		return *this;
	}
	ImgTime& operator+=(const ImgTime& other) {
		return operator+(other);
	}

private:
	TTime min, avg, max;
	size_t c = 0;
};

///////////////////////////////////////////////////////////////////////////////

TImage ProcessImage(const TImage &inImg, Options& opts, ImgTime& time)
{
	TImage labels;

	time.Reset();
	for (int i = 0; i < opts.cycles; ++i)
	{
		TTime curTime = opts.labelingAlg->Label(inImg, labels, opts.numThreads, opts.coh);		
		time.Add(curTime);
	}	

	return labels;
}

///////////////////////////////////////////////////////////////////////////////

static const std::set<std::string> extensions = {".jpg", ".bmp", ".jpeg", ".png", ".tif", ".tiff"};

std::list<std::string> FindFiles(const std::string &path)
{
	std::list<std::string> files;
	directory_iterator dirEnd;	

	for (directory_iterator itr(path); itr != dirEnd; ++itr)
	{
		auto status = itr->status();
		if (is_regular_file(itr->status()) && extensions.count(itr->path().extension().string()))
		{			
			files.push_back(itr->path().string());
		}
	}

	return files;
}

///////////////////////////////////////////////////////////////////////////////

TImage LabelsToRGB(const TImage &labels)
{
	TLabel maxLabel = 0;
	for (size_t i = 0; i < labels.total(); ++i)
	{
		TLabel lb = labels.at<TLabel>(i);
		if (lb > maxLabel) maxLabel = lb;
	}
	
	vector<uchar[3]> colorMap(maxLabel + 1);
	for (size_t i = 1; i < maxLabel + 1; ++i)
	{
		colorMap[i][0] = std::rand();
		colorMap[i][1] = std::rand();
		colorMap[i][2] = std::rand();
	}

	colorMap[0][0] = 0;
	colorMap[0][1] = 0;
	colorMap[0][2] = 0;

	TImage rgb(labels.rows, labels.cols, CV_8UC3);

	for (size_t i = 0; i < labels.total(); ++i)
	{			
		rgb.data[3 * i]     = colorMap[labels.at<TLabel>(i)][0];
		rgb.data[3 * i + 1] = colorMap[labels.at<TLabel>(i)][1];
		rgb.data[3 * i + 2] = colorMap[labels.at<TLabel>(i)][2];
	}

	return rgb;
}

///////////////////////////////////////////////////////////////////////////////

void ProcessImages(Options &opts)
{
	auto imgs = FindFiles(opts.inPath);

	size_t count = 0;
	ImgTime time;

	bool wantWrite = is_directory(opts.outPath);

	for (auto fName: imgs)
	{
		std::string fileName(path(fName).filename().string());

		TImage img = cv::imread(fName);		

		cout << "Processing image " << ++count << "/" << imgs.size() << " (" << fileName.c_str() << ")";// \n";

		ImgTime imgTime;			
		img = ProcessImage(img, opts, imgTime);

		if(wantWrite)
			cv::imwrite(opts.outPath + "/" + fileName, LabelsToRGB(img));

		time += imgTime;

		cout << " " << static_cast<float>(imgTime.Avg()) / 1000 << " ms\n";
	}

	cout << "\nMin processing time: " << static_cast<float>(time.Min()) / 1000 << " ms\n";
	cout << "Avg processing time: " << static_cast<float>(time.Avg()) / 1000 << " ms\n";	
	cout << "Max processing time: " << static_cast<float>(time.Max()) / 1000 << " ms\n";
}

///////////////////////////////////////////////////////////////////////////////

void PrintHelp(void)
{
	cout << "Usage: labeling [options]\n\n"
			"Options:\n"
			"  -i <input_path>: Input file or path\n"
			"  -o <out_path>  : Output path\n"
			"  -a <algorithm> : Labeling algorithm:\n"
			"                     bin        : Binarization\n"
			"                     he-run     : Run labeling (He)\n"
			"                     gr-block   : Block labeling (Grana)\n"
			"                     ocv        : OpenCV labeling\n"
			"                     omp-lbeq   : OpenMP Label equivalence\n"		
			"                     omp-lbeq2x : OpenMP Label equivalence for 2x2 blocks\n"
			"                     omp-runeq  : OpenMP Run equivalence\n"			
			"                     ocl-bin    : OpenCL Binarization\n"
			"                     ocl-lbeq   : OpenCL Label equivalence\n"
			"                     ocl-lbeq2x : OpenCL Label equivalence for 2x2 blocks\n"
			"                     ocl-runeq  : OpenCL Run equivalence\n"
			"  -g             : Use GPU for OpenCL (CPU otherwise)\n"
			"  -j <threads>   : Set numer of parallel threads\n"
			"  -l <cycles>    : Set numer of cycles for each image (1 default)\n"
			"  -c <connect>	  : Set connectivity (4 or 8)\n"
			"  -h             : Print this help\n\n";
}

///////////////////////////////////////////////////////////////////////////////

std::shared_ptr<ILabeling> SetLabelingAlg(const std::string &algName, bool useGPU)
{
	if (algName == "bin")
		return std::make_shared<TBinLabeling>();
	if (algName == "he-run")
		return std::make_shared<TRunLabeling>();
	if (algName == "ocl-bin")
		return std::make_shared<TOCLBinLabeling>(useGPU);
	if (algName == "ocl-lbeq")
		return std::make_shared<TOCLLabelDistribution>(useGPU);
	if (algName == "ocl-lbeq2x")
		return std::make_shared<TOCLLabelEquivalenceX2>(useGPU);
	if (algName == "ocl-runeq")
		return std::make_shared<TOCLRunEquivLabeling>(useGPU);
	if (algName == "ocv")
		return std::make_shared<TOpenCVLabeling>();
	if (algName == "gr-block")
		return std::make_shared<TBlockGranaLabeling>();
	if (algName == "omp-lbeq")
		return std::make_shared<TLabelDistribution>();
	if (algName == "omp-runeq")
		return std::make_shared<TRunEqivLabeling>();	
	if (algName == "omp-lbeq2x")
		return std::make_shared<TLabelEquivalenceX2>();

	PrintHelp();
	throw std::exception("No labeling algorithm specified");
}

///////////////////////////////////////////////////////////////////////////////

Options ParseInput(int argc, char** argv)
{
	Options opts;
	std::string algName;

	auto ReadData = [&](uint &i) -> std::string
	{
		if (i + 1 < argc)
		{
			return std::string(argv[++i]);
		}
		else
		{
			std::stringstream msg;
			msg << "Wrong input parameters (no data following the key " << argv[i++] << ")";

			throw std::exception(msg.str().c_str());
		}
	};

	// Parsing
	for (uint i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-i")) { opts.inPath = ReadData(i); continue; }
		if (!strcmp(argv[i], "-a")) { algName = ReadData(i); continue; }
		if (!strcmp(argv[i], "-o")) { opts.outPath = ReadData(i); continue; }
		if (!strcmp(argv[i], "-j")) { opts.numThreads = std::stoi(ReadData(i)); continue; }
		if (!strcmp(argv[i], "-g")) { opts.useGPU = true; continue; }
		if (!strcmp(argv[i], "-l")) { opts.cycles = std::stoi(ReadData(i)); continue; }
		if (!strcmp(argv[i], "-c")) { opts.coh = ReadData(i) == "4" ? COH_4 : COH_8; continue; }
		if (!strcmp(argv[i], "-h")) { PrintHelp(); opts.quickExit = true; return opts; }

		std::stringstream msg;
		msg << "Wrong input parameters (unknown key " << argv[i] << ")";

		throw std::exception(msg.str().c_str());
	}

	opts.labelingAlg = SetLabelingAlg(algName, opts.useGPU);

	return opts;
}

///////////////////////////////////////////////////////////////////////////////

void Run(Options &opts)
{
	if (opts.quickExit) return;

	if (is_directory(opts.inPath))
	{
		ProcessImages(opts);
	}
	else if (exists(opts.inPath))
	{
		std::string fileName(path(opts.inPath).filename().string());

		ImgTime time;
		TImage im = ProcessImage(cv::imread(opts.inPath), opts, time);

		cout << "Image: " << fileName.c_str() << "\nProcessing time: " << float(time.Avg()) / 1000 << " ms\n";			

		if (is_directory(opts.outPath))
		{
			cv::imwrite(opts.outPath + "/" + fileName, LabelsToRGB(im));
		}
	}
	else
	{
		PrintHelp();
		throw std::exception("Wrong input parameters");
	}
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	try
	{
		Run(ParseInput(argc, argv));
	}
	catch (std::exception e)
	{
		cout << "Error: " << e.what() << "\n\n";
	}
	catch (...)
	{
		cout << "Error: Unknown exception\n\n";
	}
	
	return 0;
}

///////////////////////////////////////////////////////////////////////////////