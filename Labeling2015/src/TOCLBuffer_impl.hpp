// OpenCL buffer
// Copyright (c) by Sergey Zavalishin 2015

#ifndef TOCL_BUFFER_IMPL_HPP_
#define TOCL_BUFFER_IMPL_HPP_

#include "LabelingTools.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace LabelingTools
{

	template<typename T>
		TOCLBuffer<T>::TOCLBuffer(const IOCLLabeling &ownerClass, TOCLBufferType bufType, size_t dataSize)
			: owner(ownerClass),
			  hostBuf(dataSize),
			  wantUpdate(true),
			  size(dataSize),
			  buffer(deviceBuf)
		{
			memFlags =
				bufType == READ_ONLY  ? CL_MEM_READ_ONLY  :
				bufType == WRITE_ONLY ? CL_MEM_WRITE_ONLY :
				/* default */			CL_MEM_READ_WRITE;

			CreateDeviceBuffer();
		}

	///////////////////////////////////////////////////////////////////////////////
		
	template<typename T>
		TOCLBuffer<T>::~TOCLBuffer(void)
		{
			DeleteDeviceBuffer();
		}

	///////////////////////////////////////////////////////////////////////////////

	template<typename T>
		void TOCLBuffer<T>::Push(void)
		{
			if (wantUpdate)
				UpdateDeviceBuffer();
		}

	///////////////////////////////////////////////////////////////////////////////
		
	template<typename T>
		void TOCLBuffer<T>::Pull(void)
		{
			UpdateHostBuffer();
		}

	///////////////////////////////////////////////////////////////////////////////

	template<typename T>
		vector<T>& TOCLBuffer<T>::Buffer(void)
		{
			wantUpdate = true;
			return hostBuf;
		}		

	///////////////////////////////////////////////////////////////////////////////
	
	template<typename T>
		void TOCLBuffer<T>::UpdateDeviceBuffer(void)
		{
			// Pre Conditions
			if (!wantUpdate)
				return;
			if (!owner.Initialized)		
				throw(std::exception("TOCLBuffer::UpdateDeviceBuffer : Buffer owner is not initialized"));
			if (!isInitialized)
				CreateDeviceBuffer();			
			if (size != hostBuf.size())
				ResizeDeviceBuffer();

			// Actual Code
			clErrorContext = clEnqueueWriteBuffer(owner.State.queue, deviceBuf, CL_TRUE, 0,
				size * sizeof(T), &hostBuf[0], 0, NULL, NULL);


			// Post Conditions
			THROW_IF_OCL(clErrorContext, "TOCLBuffer::UpdateDeviceBuffer")
		}
		
	///////////////////////////////////////////////////////////////////////////////

	template<typename T>
		void TOCLBuffer<T>::UpdateHostBuffer(void)
		{			
			// Pre Conditions
			if (!owner.Initialized)
				throw(std::exception("TOCLBuffer::UpdateHostBuffer : Buffer owner is not initialized"));
			if (!isInitialized)
				CreateDeviceBuffer();
			if (size != hostBuf.size())
				ResizeDeviceBuffer();

			// Actual Code
			clErrorContext = clEnqueueReadBuffer(owner.State.queue, deviceBuf, CL_TRUE, 0,
				size * sizeof(T), &hostBuf[0], 0, NULL, NULL);

			// Post Conditions
			THROW_IF_OCL(clErrorContext, "TOCLBuffer::UpdateHostBuffer")
		}
		
	///////////////////////////////////////////////////////////////////////////////

	template<typename T>
		void TOCLBuffer<T>::CreateDeviceBuffer(void)
		{
			// Pre Conditions
			if (!owner.Initialized)
				throw(std::exception("TOCLBuffer::UpdateHostBuffer : Buffer owner is not initialized"));

			// Actual Code
			size = hostBuf.size();

			deviceBuf = clCreateBuffer(owner.State.context, memFlags, size * sizeof(T),
				NULL, &clErrorContext);
			
			THROW_IF_OCL(clErrorContext, "TOCLBuffer::CreateDeviceBuffer")

			// Post Conditions
			wantUpdate = true;
			isInitialized = true;
		}

	///////////////////////////////////////////////////////////////////////////////

	template<typename T>
		void TOCLBuffer<T>::DeleteDeviceBuffer(void)
		{
			// Pre Conditions
			if (!isInitialized) 
				return;

			// Actual Code
			clErrorContext = clReleaseMemObject(deviceBuf);

			THROW_IF_OCL(clErrorContext, "TOCLBuffer::DeleteDeviceBuffer")
		}

	///////////////////////////////////////////////////////////////////////////////

	template<typename T>
		void TOCLBuffer<T>::ResizeDeviceBuffer(void)
		{			
			// Pre Conditions
			if (!isInitialized)		 
				CreateDeviceBuffer();
			if (!owner.Initialized)
				throw(std::exception("TOCLBuffer::ResizeDeviceBuffer : Buffer owner is not initialized"));

			// Actual Code
			DeleteDeviceBuffer();
			CreateDeviceBuffer();

			// Post Conditions
			wantUpdate = true;			
		}

	///////////////////////////////////////////////////////////////////////////////

} /* LabelingTools */

#endif /* TOCL_BUFFER_IMPL_HPP_ */