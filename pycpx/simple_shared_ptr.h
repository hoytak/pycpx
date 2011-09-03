#ifndef _SIMPLE_SHARED_PTR_H_
#define _SIMPLE_SHARED_PTR_H_

template <typename T> class SharedPointer 
{
private:
    T* __restrict__ _data;
    size_t *_ref_count;

    inline void decRef()
    {
	if( --(*_ref_count) == 0)
	{
	    if(_data != NULL)
		delete _data;

	    delete _ref_count;
	}
    }

public:
    SharedPointer() 
    : _data(NULL), _ref_count(new size_t) 
    {
	(*_ref_count) = 1;
    }
    
    SharedPointer(T* value) 
    : _data(value), _ref_count(new size_t)
    {
	(*_ref_count) = 1;
    }
    
    SharedPointer(const SharedPointer<T>& sp) 
    : _data(sp._data), _ref_count(sp._ref_count)
    {
	++(*_ref_count);
    }
    
    ~SharedPointer()
    {
	decRef();
    }

    inline T& operator*() const { return *_data; }
    inline T* operator->() const { return _data; }

    SharedPointer<T>& operator=(const SharedPointer<T>& sp)
    {
	if (this != &sp) // Avoid self assignment
	{
	    decRef();

	    _data = sp._data;
	    _ref_count = sp._ref_count;
	    ++(*_ref_count);
	}
	return *this;
    }
    
    bool operator==(void* other) const
    {
	return _data == other;
    }

    bool operator!=(void* other) const
    {
	return _data != other;
    }
};

#endif /* _SIMPLE_SHARED_PTR_H_ */
