//Labeling Tools
//Copyright (c) by Sergey Zavalishin 2010-2015
//
//Contains main classes for basic image labeling.

#ifndef LABELING_TOOLS_HPP_
#define LABELING_TOOLS_HPP_

#include <vector>
#include <omp.h>
#include <opencv2/core/core.hpp>

#include "stopwatch_win.h"

#include "CLUtils.h"

///////////////////////////////////////////////////////////////////////////////

#define THROW_IF(EXPR, MSG) if(EXPR) { throw std::exception(#MSG); }
#define THROW_IF_OCL(ERR_CODE, METHOD) \
	if (ERR_CODE != CL_SUCCESS) \
	{ \
		std::stringstream str; \
		str << #METHOD << " : OpenCL error (error code \""<< ERR_CODE << "\")"; \
		throw std::exception(str.str().c_str()); \
	}

///////////////////////////////////////////////////////////////////////////////

using namespace std;

///////////////////////////////////////////////////////////////////////////////

namespace LabelingTools
{

	typedef uchar			TPixel;	
	typedef uint			TLabel;	
	typedef vector<TPixel>	TPixels;
	typedef uint			TTime;
	typedef uint			TColor;
	typedef cv::Mat			TImage;
	
	typedef enum TCoherence
	{
		COH_4,
		COH_8,
		COH_DEFAULT
	};

	const char MAX_THREADS = 0;

	///////////////////////////////////////////////////////////////////////////////
	// ILabeling definition (basic labeling algorithm class)
	///////////////////////////////////////////////////////////////////////////////

	class ILabeling
	{
	public:		
		virtual ~ILabeling(void) = default;

		// Call to start labeling
		virtual TTime Label(const TImage& pixels, TImage& labels, char threads = MAX_THREADS, TCoherence coh = TCoherence::COH_DEFAULT);

		TImage RGB2Gray(const TImage& img) const;

	protected:
		StopWatchWin watch_;

		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) = 0; // Labeling itself
		void SetupThreads(char threadNum); //threads setup
	};

	///////////////////////////////////////////////////////////////////////////////
	// IOCLLabeling definition (basic OpenCL labeling algorithm class)
	///////////////////////////////////////////////////////////////////////////////

	// Forward declaration
	template <typename T> class TOCLBuffer;

	// OCL Labeling error
	typedef enum TOCLLabelingError
	{
		OK = 0,
		WRONG_INPUT_PARAMS,
		CANT_CREATE_CONTEXT,
		CANT_CREATE_COMMAND_QUEUE,
		CANT_OPEN_KERNEL_SOURCE_FILE,
		CANT_CREATE_PROGRAM_WITH_SOURCE,
		CANT_BUILD_PROGRAM,
		CANT_GET_DEVICE_INFO,
		OWNER_IS_NOT_INITIALIZED,
		OCL_ERROR,
		OCL_MAX_ERROR
	};

	class IOCLLabeling : public ILabeling
	{
	public:
		const bool &Initialized;	// Shows if device is initialized
		const clState &State;		// OpenCL state structure

		// Opens device with specified algorithm source
		void Init(cl_device_type deviceType, const std::string& buildParams, const std::string& srcFileName);

		// Call to start labeling
		virtual TTime Label(const TImage& pixels, TImage& labels, char threads = MAX_THREADS, TCoherence coh = TCoherence::COH_DEFAULT) override;
		
		// Destructor
		~IOCLLabeling(void);

	protected:
		clState OCLState;	// OCL state structure
		bool isInitialized;	// Shows if device is initialized

		IOCLLabeling(void);

		virtual void TerminateOCL(void);

		// Write your OCL labeling code here
		virtual void DoOCLLabel(TOCLBuffer<TPixel> &pixels, TOCLBuffer<TLabel> &labels, unsigned int imgWidth, 
								unsigned int imgHeight, TCoherence Coherence) = 0;

		// Write your kernel initialization code here
		virtual void InitKernels(void) = 0;

		// Write your kernel finalization code here
		virtual void FreeKernels(void) {}; // Used in destructor, that's why non-pure virtual

	private:				
		IOCLLabeling(const IOCLLabeling&) = delete;
		IOCLLabeling& operator= (const IOCLLabeling&) = delete;
		virtual void DoLabel(const TImage& pixels, TImage& labels, char threads, TCoherence coh) {}; // Deprecated
	};

	///////////////////////////////////////////////////////////////////////////////
	// TOCLBuffer definition (basic OpenCL buffer class template)
	///////////////////////////////////////////////////////////////////////////////

	// Buffer type
	typedef enum TOCLBufferType 
	{ 
		READ_ONLY	= 0x1,		// Read only for device
		WRITE_ONLY	= 0x2,		// Write only for device
		READ_WRITE	= 0x3,		// Read+write for device
		MAX_BUFFER_TYPE
	};

	template <typename DataType> class TOCLBuffer final
	{
	public:		
		const IOCLLabeling &owner;	// Buffer owner
		const cl_mem &buffer;		// OpenCL buffer (pass it as kernel param)
		cl_int clErrorContext;		// Stores last OpenCL error code (if OCL_ERROR has occured)

		// Constructor
		TOCLBuffer(const IOCLLabeling &ownerClass, TOCLBufferType bufType, size_t dataSize);

		// Destructor
		~TOCLBuffer(void);

		// Uploads buffer to device
		void Push(void);

		// Downloads buffer from device
		void Pull(void);

		// Returns buffer object
		vector<DataType>& Buffer(void);

		// Direct buffer access (slow!)
		DataType& operator[] (size_t index) 
		{
			wantUpdate = true;
			return hostBuf[index];
		}

	private:
		size_t size;				// Device buffer size
		bool wantUpdate;			// Shows if device buffer need to be updated
		bool isInitialized;			// Shows if device buffer is initialized
		cl_mem_flags memFlags;		// Device memory flags

		cl_mem deviceBuf;			// Buffer device copy
		vector<DataType> hostBuf;	// Host buffer (accessible directly)

		// Updates device buffer
		void UpdateDeviceBuffer(void);

		// Updates host buffer
		void UpdateHostBuffer(void);

		// Creates device buffer
		void CreateDeviceBuffer(void);

		// Deletes device buffer
		void DeleteDeviceBuffer(void);

		// Resizes device buffer
		void ResizeDeviceBuffer(void);

		TOCLBuffer(void) = delete;
		TOCLBuffer(const TOCLBuffer<DataType>&) = delete;
		TOCLBuffer operator= (const TOCLBuffer<DataType>&) = delete;
	};

} /* LabelingTools*/

#	include "TOCLBuffer_impl.hpp" // Template implementation

#endif /* LABELING_TOOLS_HPP_ */