#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cout << "CUDA Error: " << err << " at line " << __LINE__ << " in file " << __FILE__ << std::endl;\
            std::cout<<cudaGetErrorString(err)<<std::endl;\
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                " at line " + std::to_string(__LINE__) + " in file " + __FILE__); \
        } \
    } while (0)

// 设备类型枚举
enum class MemoryLocation {
    CPU,    // 仅CPU
    GPU,    // 仅GPU
    BOTH,    // CPU和GPU都有（数据同步）
    NO       // 无
};

// 通用设备内存缓冲区：封装CPU/GPU内存管理
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : host_ptr_(nullptr), device_ptr_(nullptr), size_(0), device_id_(0),
        is_host_allocated_(false), is_device_allocated_(false) {
    }

    ~DeviceBuffer() {
        free_host();
        free_device();
    }

    // 禁止拷贝，避免双重释放（如需拷贝可实现深拷贝）
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // 移动构造/赋值
    DeviceBuffer(DeviceBuffer&& other) noexcept {
        *this = std::move(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            // 释放当前内存
            free_host();
            free_device();

            // 接管对方资源
            host_ptr_ = other.host_ptr_;
            device_ptr_ = other.device_ptr_;
            size_ = other.size_;
            device_id_ = other.device_id_;
            is_host_allocated_ = other.is_host_allocated_;
            is_device_allocated_ = other.is_device_allocated_;

            // 清空对方
            other.host_ptr_ = nullptr;
            other.device_ptr_ = nullptr;
            other.size_ = 0;
            other.is_host_allocated_ = false;
            other.is_device_allocated_ = false;
        }
        return *this;
    }

    // 分配CPU内存
    void allocate_host(size_t size) {
        free_host();
        if (size == 0) return;
        host_ptr_ = new T[size];
        memset(host_ptr_, 0, size * sizeof(T));
        size_ = size;
        is_host_allocated_ = true;
    }

    // 分配GPU内存（指定设备ID）
    void allocate_device(size_t size, int device_id = 0) {
        free_device();
        if (size == 0) return;
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        CHECK_CUDA_ERROR(cudaMalloc(&device_ptr_, size * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMemset(device_ptr_, 0, size * sizeof(T)));
        size_ = size;
        device_id_ = 0;
        is_device_allocated_ = true;
    }

    // 释放CPU内存
    void free_host() {
        if (is_host_allocated_ && host_ptr_) {
            delete[] static_cast<T*>(host_ptr_);
            host_ptr_ = nullptr;
            is_host_allocated_ = false;
        }
    }

    // 释放GPU内存
    void free_device() {
        if (is_device_allocated_ && device_ptr_) {
            CHECK_CUDA_ERROR(cudaSetDevice(0));
            CHECK_CUDA_ERROR(cudaFree(device_ptr_));
            device_ptr_ = nullptr;
            is_device_allocated_ = false;
        }
    }

    // CPU -> GPU 拷贝（自动分配GPU内存）
    void copy_host_to_device(int device_id = 0) {
        if (!is_host_allocated_) {
            throw std::runtime_error("Host memory not allocated for copy");
        }
        if (!is_device_allocated_) {
            allocate_device(size_, 0);
        }
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        CHECK_CUDA_ERROR(cudaMemcpy(device_ptr_, host_ptr_, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    // GPU -> CPU 拷贝（自动分配CPU内存）
    void copy_device_to_host() {
        if (!is_device_allocated_) {
            throw std::runtime_error("Device memory not allocated for copy");
        }
        if (!is_host_allocated_) {
            allocate_host(size_);
        }
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        CHECK_CUDA_ERROR(cudaMemcpy(host_ptr_, device_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // 获取CPU指针（类型安全）
    T* get_host_ptr() {
        return static_cast<T*>(host_ptr_);
    }

    // 获取GPU指针（类型安全）
    T* get_device_ptr() {
        return static_cast<T*>(device_ptr_);
    }

    // 获取内存大小（元素个数）
    size_t size() const { return size_; }

    // 判断数据位置
    MemoryLocation location() const {
        if (is_host_allocated_ && is_device_allocated_) return MemoryLocation::BOTH;
        if (is_host_allocated_) return MemoryLocation::CPU;
        if (is_device_allocated_) return MemoryLocation::GPU;
        return MemoryLocation::NO;
    }

    // 重置缓冲区
    void reset() {
        free_host();
        free_device();
        size_ = 0;
        device_id_ = 0;
    }

private:
    void* host_ptr_;          // CPU指针
    void* device_ptr_;        // GPU指针
    size_t size_;             // 元素个数
    int device_id_;           // GPU设备ID
    bool is_host_allocated_;  // CPU内存是否分配
    bool is_device_allocated_;// GPU内存是否分配
};