// Labeling2015.cpp : Defines the entry point for the console application.
//

#include <time.h>

#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <array>

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

	enum {OCL_NO, OCL_CPU, OCL_GPU} useOCL = OCL_NO;
	TCoherence coh = COH_DEFAULT;
	bool label3D = false;
	bool quickExit = false;
};

///////////////////////////////////////////////////////////////////////////////

struct ImgTime
{	
	void Reset(void) { min = UINT_MAX; c = avg = max = 0; }	
	
	TTime Min(void) const { return min != UINT_MAX ? min : 0; }
	TTime Max(void) const { return max; }
	TTime Avg(void) const { return c ? static_cast<float>(avg) / c : 0; }
	
	void Add(TTime t) {
		if (UINT_MAX - avg < t) CompressData();
		if (min > t) min = t;
		if (max < t) max = t;
		avg += t;
		++c;
	}
	
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

	ImgTime(void) { Reset(); }

private:
	TTime min, avg, max;
	size_t c = 0;

	void CompressData(void) {
		avg /= c;
		c = 1;
	}
};

///////////////////////////////////////////////////////////////////////////////

struct Algs
{
	const std::string descr;
	const std::function<std::shared_ptr<ILabeling>(void)> cpu ;
	const std::function<std::shared_ptr<IOCLLabeling>(bool)> ocl;
	const std::function<std::shared_ptr<ILabeling>(void)> cpu3d ;
	const std::function<std::shared_ptr<IOCLLabeling>(bool)> ocl3d;
};

static std::map<std::string, Algs> ALG_LIST;

///////////////////////////////////////////////////////////////////////////////

void InitAlgs(void)
{	
	ALG_LIST.emplace(std::string("bin"), Algs{ "Simple binarization", 
												&std::make_shared<TBinLabeling>, 
												&std::make_shared<TOCLBinLabeling, bool>, 
												nullptr, nullptr });
	ALG_LIST.emplace(std::string("he-run"), Algs{ "Run labeling by He et.al.", 
												&std::make_shared<TRunLabeling>, 
												nullptr, nullptr, nullptr });
	ALG_LIST.emplace(std::string("gr-block"), Algs{ "Block labeling by Grana et.al.", 
												&std::make_shared<TBlockGranaLabeling>, 
												nullptr, nullptr, nullptr });
	ALG_LIST.emplace(std::string("ocv"), Algs{ "OpenCV labeling", 
												&std::make_shared<TOpenCVLabeling>, 
												nullptr, nullptr, nullptr });
	ALG_LIST.emplace(std::string("lbeq"), Algs{ "Label equivalence by Kalentev et.al.", 
												&std::make_shared<TLabelDistribution>, 
												&std::make_shared<TOCLLabelDistribution, bool>, 
												nullptr, nullptr });
	ALG_LIST.emplace(std::string("bleq"), Algs{ "Block equivalence by Zavalishin et.al.", 
												&std::make_shared<TLabelEquivalenceX2>, 
												std::make_shared<TOCLLabelEquivalenceX2, bool>, 
												nullptr, 
												nullptr/*std::make_shared<TOCLLabelEquivalenceX2, bool>*/ });
	ALG_LIST.emplace(std::string("runeq"), Algs{ "Run equivalence by Zavalishin et.al.", 
												&std::make_shared<TRunEqivLabeling>, 
												&std::make_shared<TOCLRunEquivLabeling, bool>, 
												nullptr, nullptr });
}

///////////////////////////////////////////////////////////////////////////////

static const std::set<std::string> extensions = { ".jpg", ".bmp", ".jpeg", ".png", ".tif", ".tiff" };

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

TImage ProcessImage(const TImage &inImg, const Options& opts, ImgTime& time)
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

TImage Read3DImage(const std::string &inPath)
{
	if (is_directory(inPath))
	{
		auto files = FindFiles(inPath);
		
		TImage outIm, curIm;
		std::vector<cv::Range> ranges(3);
		int curPlane = 0;

		for (auto fName : files)
		{
			std::string fileName(path(fName).filename().string());
			curIm = cv::imread(fileName, cv::IMREAD_GRAYSCALE);

			if (outIm.empty()) 
			{
				int sz[] = {curIm.rows, curIm.cols, files.size()};
				outIm = TImage(3, sz, CV_8U);

				ranges[0].start = ranges[1].start = 0;
				ranges[0].end = curIm.rows;
				ranges[1].end = curIm.cols;
			}
			
			if (curIm.rows != outIm.rows || curIm.cols != outIm.cols) {
				throw std::exception("Cannot read 3D image: slice sizes do not match");
			}

			ranges[2].start = ranges[2].end = curPlane++;
			curIm.copyTo(outIm(ranges.data()));
		}

		return outIm;
	}

	return TImage();
}

///////////////////////////////////////////////////////////////////////////////

TImage Process3DImage(const TImage &inImg, const Options& opts, ImgTime& time)
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

TLabel GetMaxLabel(const TImage &labels)
{
	TLabel maxLabel = 0;
	for (size_t i = 0; i < labels.total(); ++i)
	{
		TLabel lb = labels.at<TLabel>(i);
		if (lb > maxLabel) maxLabel = lb;
	}

	return maxLabel;
}

///////////////////////////////////////////////////////////////////////////////

typedef std::vector<std::array<uchar, 3>> ColorMap;

void UpdateColorMap(ColorMap &colorMap, TLabel maxLabel)
{	
	size_t oldSize = colorMap.size();
	colorMap.resize(maxLabel + 1);

	for (size_t i = oldSize; i < maxLabel + 1; ++i)
	{
		colorMap[i][0] = std::rand();
		colorMap[i][1] = std::rand();
		colorMap[i][2] = std::rand();
	}

	colorMap[0][0] = 0;
	colorMap[0][1] = 0;
	colorMap[0][2] = 0;	
}

///////////////////////////////////////////////////////////////////////////////

TImage LabelsToRGB(const TImage &labels, ColorMap& colorMap = ColorMap())
{	
	TLabel maxLabel = GetMaxLabel(labels);

	if (colorMap.size() < maxLabel + 1)
		UpdateColorMap(colorMap, maxLabel);

	TImage rgb(labels.rows, labels.cols, CV_8UC3);

	for (size_t i = 0; i < labels.total(); ++i)
	{
		rgb.data[3 * i] = colorMap[labels.at<TLabel>(i)][0];
		rgb.data[3 * i + 1] = colorMap[labels.at<TLabel>(i)][1];
		rgb.data[3 * i + 2] = colorMap[labels.at<TLabel>(i)][2];
	}

	return rgb;
}

///////////////////////////////////////////////////////////////////////////////

void Write3DLabels(const TImage &labels, const std::string outPath)
{
	create_directories(outPath);

	std::vector<cv::Range> ranges(3);
	ranges[0].start = ranges[1].start = 0;
	ranges[0].end = labels.rows;
	ranges[1].end = labels.cols;

	ColorMap colorMap;

	for (int i = 0; i < labels.size[2]; ++i)
	{
		ranges[2].start = ranges[2].end = i;
		cv::imwrite(outPath + '/' + std::to_string(i) + ".png", LabelsToRGB(labels(ranges.data()), colorMap));
	}
}

///////////////////////////////////////////////////////////////////////////////

void ProcessImages(const Options &opts)
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
			"  -i <input_path> : Input file or path\n"
			"  -o <out_path>   : Output path\n"
			"  -a <algorithm>  : Labeling algorithm:\n";

	for (auto alg : ALG_LIST) {
		cout << setw(12) << ' ' << setw(9) << left << alg.first << ": " << alg.second.descr
			 << " (" 
			 << (alg.second.cpu != nullptr ? "CPU" : "")
			 << (alg.second.cpu3d != nullptr ? "+3D" : "")
			 << (alg.second.ocl != nullptr ? " OCL" : "")
			 << (alg.second.ocl3d != nullptr ? "+3D" : "")
			 << ")\n";
	}

	cout << "  -3           : Theat input sequence as a single 3D image\n"
			"  -g           : Run algorithm in OpenCL mode on GPU (if available)\n"
			"  -u           : Run algorithm in OpenCL mode on CPU (if available)\n"
			"  -j <threads> : Set numer of parallel threads for OpenMP (default 0)\n"
			"  -l <cycles>  : Set numer of cycles for each image (default 1)\n"
			"  -c <connect> : Set connectivity (4 or 8 [default])\n"
			"  -h           : Print this help\n\n";
}

///////////////////////////////////////////////////////////////////////////////

std::shared_ptr<ILabeling> SetLabelingAlg(const std::string &algName, const Options& opts)
{
	bool useOCL = opts.useOCL != Options::OCL_NO;
	bool useGPU = opts.useOCL == Options::OCL_GPU;
	bool label3D = opts.label3D;

	auto algCreator = ALG_LIST.find(algName);
	std::function<std::shared_ptr<ILabeling>(void)> creatorFunc;
	std::function<std::shared_ptr<IOCLLabeling>(bool)> oclCreatorFunc;

	if (algCreator != ALG_LIST.end())
	{
		if (!label3D)
			if (!useOCL)
				creatorFunc = algCreator->second.cpu;
			else
				oclCreatorFunc = algCreator->second.ocl;
		else
			if (!useOCL)
				creatorFunc = algCreator->second.cpu3d;
			else
				oclCreatorFunc = algCreator->second.ocl3d;

		if (!useOCL && creatorFunc != nullptr)
			return creatorFunc();
		else if (oclCreatorFunc != nullptr)
			return oclCreatorFunc(useGPU);

		throw std::exception("Choosen algorithm doesn't support specified capabilities");
	}
	
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
		if (!strcmp(argv[i], "-o")) { opts.outPath = ReadData(i); continue; }

		if (!strcmp(argv[i], "-a")) { algName = ReadData(i); continue; }			
		
		if (!strcmp(argv[i], "-3")) { opts.label3D = true; continue; }		
		if (!strcmp(argv[i], "-g")) { opts.useOCL = Options::OCL_GPU; continue; }
		if (!strcmp(argv[i], "-u")) { opts.useOCL = Options::OCL_CPU; continue; }		
		
		if (!strcmp(argv[i], "-j")) { opts.numThreads = std::stoi(ReadData(i)); continue; }
		if (!strcmp(argv[i], "-l")) { opts.cycles = std::stoi(ReadData(i)); continue; }
		if (!strcmp(argv[i], "-c")) { opts.coh = ReadData(i) == "4" ? COH_4 : COH_8; continue; }
		
		if (!strcmp(argv[i], "-h")) { PrintHelp(); opts.quickExit = true; return opts; }

		std::stringstream msg;
		msg << "Wrong input parameters (unknown key " << argv[i] << ")";

		throw std::exception(msg.str().c_str());
	}

	opts.labelingAlg = SetLabelingAlg(algName, opts);

	return opts;
}

///////////////////////////////////////////////////////////////////////////////

void Process2DImages(const Options &opts)
{
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
		throw std::exception("Wrong input path");
	}
}

///////////////////////////////////////////////////////////////////////////////

void Process3DImages(const Options &opts)
{
	if (is_directory(opts.inPath))
	{
		ImgTime time;
		TImage im = Process3DImage(Read3DImage(opts.inPath), opts, time);

		cout << "Image: " << opts.inPath << "\nProcessing time: " << float(time.Avg()) / 1000 << " ms\n";

		if (is_directory(opts.outPath))
		{
			Write3DLabels(im, opts.outPath);
		}
	}
	else
	{
		PrintHelp();
		throw std::exception("Wrong input path (assumed directory with 3D image slices)");
	}
}

///////////////////////////////////////////////////////////////////////////////

void Run(const Options &opts)
{
	if (opts.quickExit) return;

	if (!opts.label3D)
		Process2DImages(opts);
	else
		Process3DImages(opts);
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	try
	{
		InitAlgs();
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