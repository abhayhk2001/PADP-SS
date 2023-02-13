
#include "cm.h"
#include "zone_map.h"
#include "moderngpu/src/moderngpu/kernel_reduce.hxx"
#include "moderngpu/src/moderngpu/kernel_segreduce.hxx"


using namespace mgpu;
using namespace thrust::placeholders;

vector<void*> alloced_mem;

template<typename T>
struct distinct : public binary_function<T,T,T>
{
    __host__ __device__ T operator()(const T &lhs, const T &rhs) const {
        return lhs != rhs;
    }
};



struct gpu_getyear
{
    const int_type *source;
    int_type *dest;

    gpu_getyear(const int_type *_source, int_type *_dest):
        source(_source), dest(_dest) {}
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {

        unsigned long long int sec;
        uint quadricentennials, centennials, quadrennials, annuals/*1-ennial?*/;
        uint year, leap;
        uint yday;
        uint month, mday;
        const uint daysSinceJan1st[2][13]=
        {
            {0,31,59,90,120,151,181,212,243,273,304,334,365}, // 365 days, non-leap
            {0,31,60,91,121,152,182,213,244,274,305,335,366}  // 366 days, leap
        };
        unsigned long long int SecondsSinceEpoch = source[i]/1000;
        sec = SecondsSinceEpoch + 11644473600;

        //wday = (uint)((sec / 86400 + 1) % 7); // day of week
        quadricentennials = (uint)(sec / 12622780800ULL); // 400*365.2425*24*3600
        sec %= 12622780800ULL;

        centennials = (uint)(sec / 3155673600ULL); // 100*(365+24/100)*24*3600
        if (centennials > 3)
        {
            centennials = 3;
        }
        sec -= centennials * 3155673600ULL;

        quadrennials = (uint)(sec / 126230400); // 4*(365+1/4)*24*3600
        if (quadrennials > 24)
        {
            quadrennials = 24;
        }
        sec -= quadrennials * 126230400ULL;

        annuals = (uint)(sec / 31536000); // 365*24*3600
        if (annuals > 3)
        {
            annuals = 3;
        }
        sec -= annuals * 31536000ULL;

        year = 1601 + quadricentennials * 400 + centennials * 100 + quadrennials * 4 + annuals;
        leap = !(year % 4) && (year % 100 || !(year % 400));

        // Calculate the day of the year and the time
        yday = sec / 86400;
        sec %= 86400;
        //hour = sec / 3600;
        sec %= 3600;
        //min = sec / 60;
        sec %= 60;

        // Calculate the month
        for (mday = month = 1; month < 13; month++)
        {
            if (yday < daysSinceJan1st[leap][month])
            {
                mday += yday - daysSinceJan1st[leap][month - 1];
                break;
            }
        }
        dest[i] = year;
    }
};

struct gpu_getmonth
{
    const int_type *source;
    int_type *dest;

    gpu_getmonth(const int_type *_source, int_type *_dest):
        source(_source), dest(_dest) {}
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {

        unsigned long long int sec;
        uint quadricentennials, centennials, quadrennials, annuals/*1-ennial?*/;
        uint year, leap;
        uint yday;
        uint month, mday;
        const uint daysSinceJan1st[2][13]=
        {
            {0,31,59,90,120,151,181,212,243,273,304,334,365}, // 365 days, non-leap
            {0,31,60,91,121,152,182,213,244,274,305,335,366}  // 366 days, leap
        };
        unsigned long long int SecondsSinceEpoch = source[i]/1000;
        sec = SecondsSinceEpoch + 11644473600;

        //wday = (uint)((sec / 86400 + 1) % 7); // day of week
        quadricentennials = (uint)(sec / 12622780800ULL); // 400*365.2425*24*3600
        sec %= 12622780800ULL;

        centennials = (uint)(sec / 3155673600ULL); // 100*(365+24/100)*24*3600
        if (centennials > 3)
        {
            centennials = 3;
        }
        sec -= centennials * 3155673600ULL;

        quadrennials = (uint)(sec / 126230400); // 4*(365+1/4)*24*3600
        if (quadrennials > 24)
        {
            quadrennials = 24;
        }
        sec -= quadrennials * 126230400ULL;

        annuals = (uint)(sec / 31536000); // 365*24*3600
        if (annuals > 3)
        {
            annuals = 3;
        }
        sec -= annuals * 31536000ULL;

        year = 1601 + quadricentennials * 400 + centennials * 100 + quadrennials * 4 + annuals;
        leap = !(year % 4) && (year % 100 || !(year % 400));

        // Calculate the day of the year and the time
        yday = sec / 86400;
        sec %= 86400;
        //hour = sec / 3600;
        sec %= 3600;
        //min = sec / 60;
        sec %= 60;

        // Calculate the month
        for (mday = month = 1; month < 13; month++)
        {
            if (yday < daysSinceJan1st[leap][month])
            {
                mday += yday - daysSinceJan1st[leap][month - 1];
                break;
            }
        }
        dest[i] = year*100+month;
    }
};


struct gpu_getday
{
    const int_type *source;
    int_type *dest;

    gpu_getday(const int_type *_source, int_type *_dest):
        source(_source), dest(_dest) {}
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {

        unsigned long long int sec;
        uint quadricentennials, centennials, quadrennials, annuals/*1-ennial?*/;
        uint year, leap;
        uint yday;
        uint month, mday;
        const uint daysSinceJan1st[2][13]=
        {
            {0,31,59,90,120,151,181,212,243,273,304,334,365}, // 365 days, non-leap
            {0,31,60,91,121,152,182,213,244,274,305,335,366}  // 366 days, leap
        };
        unsigned long long int SecondsSinceEpoch = source[i]/1000;
        sec = SecondsSinceEpoch + 11644473600;

        //wday = (uint)((sec / 86400 + 1) % 7); // day of week
        quadricentennials = (uint)(sec / 12622780800ULL); // 400*365.2425*24*3600
        sec %= 12622780800ULL;

        centennials = (uint)(sec / 3155673600ULL); // 100*(365+24/100)*24*3600
        if (centennials > 3)
        {
            centennials = 3;
        }
        sec -= centennials * 3155673600ULL;

        quadrennials = (uint)(sec / 126230400); // 4*(365+1/4)*24*3600
        if (quadrennials > 24)
        {
            quadrennials = 24;
        }
        sec -= quadrennials * 126230400ULL;

        annuals = (uint)(sec / 31536000); // 365*24*3600
        if (annuals > 3)
        {
            annuals = 3;
        }
        sec -= annuals * 31536000ULL;

        year = 1601 + quadricentennials * 400 + centennials * 100 + quadrennials * 4 + annuals;
        leap = !(year % 4) && (year % 100 || !(year % 400));

        // Calculate the day of the year and the time
        yday = sec / 86400;
        sec %= 86400;
        //hour = sec / 3600;
        sec %= 3600;
        //min = sec / 60;
        sec %= 60;

        // Calculate the month
        for (mday = month = 1; month < 13; month++)
        {
            if (yday < daysSinceJan1st[leap][month])
            {
                mday += yday - daysSinceJan1st[leap][month - 1];
                break;
            }
        }
        dest[i] = year*10000+month*100+mday;
    }
};

void make_calc_columns(queue<string> op_type, queue<string> op_value, CudaSet* a, set<string>& order_field_names)
{
	string ss, s1_val;
    stack<string> exe_type, exe_value;
	string op_t, op_v;
	unsigned int bits;
	
	for(int i=0; !op_type.empty(); ++i, op_type.pop()) {
        ss = op_type.front();
		
		if (ss.compare("NAME") == 0) {
			if(!op_value.empty()) {
				exe_value.push(op_value.front());
				op_value.pop();	
			};
                }
		else if (ss.compare("CAST") == 0 || ss.compare("YEAR") == 0) {
			op_v = exe_value.top();
			exe_value.pop();
			op_t = ss;			
		}
		else if (ss.compare("emit sel_name") == 0) {
			if(!op_t.empty()) {
				
				if(cpy_bits.empty())
					bits = 0;
				else	
					bits = cpy_bits[op_v];			
			
				if(order_field_names.find(op_value.front()) == order_field_names.end()) {
					order_field_names.insert(op_value.front());
					order_field_names.erase(op_v);
				};	

				a->columnNames.push_back(op_value.front());
				a->cols[a->cols.size()+1] = op_value.front();
				a->type[op_value.front()] = 0;
				a->decimal[op_value.front()] = 0;
				a->decimal_zeroes[op_value.front()] = 0;
				
				
				a->h_columns_int[op_value.front()] = thrust::host_vector<int_type, pinned_allocator<int_type> >(a->mRecCount);
				a->d_columns_int[op_value.front()] = thrust::device_vector<int_type>(a->mRecCount);
				if (op_t.compare("CAST") == 0) {
					cpy_bits[op_value.front()] = bits;
					
					cpy_init_val[op_value.front()] = cpy_init_val[op_v]/100;
					else
						thrust::transform(a->d_columns_int[op_v].begin(), a->d_columns_int[op_v].begin() + a->mRecCount, a->d_columns_int[op_value.front()].begin(), _1/100);					
				}
				else {
					cpy_init_val[op_value.front()] = 0;
					cpy_bits[op_value.front()] = 0;
					else				
						thrust::transform(a->d_columns_int[op_v].begin(), a->d_columns_int[op_v].begin() + a->mRecCount, thrust::make_constant_iterator(10000), a->d_columns_int[op_value.front()].begin(), thrust::divides<int_type>());															
				};
				op_t.clear();
			};	
			op_value.pop();		
		}
		else  if (ss.compare("MUL") == 0  || ss.compare("ADD") == 0 || ss.compare("DIV") == 0 || ss.compare("MINUS") == 0) {
			if(!exe_value.empty())
			    exe_value.pop();
			if(!exe_value.empty())
                            exe_value.pop();
		};	
		
	};
}

bool select(queue<string> op_type, queue<string> op_value, queue<int_type> op_nums, queue<float_type> op_nums_f, queue<unsigned int> op_nums_precision, CudaSet* a,
            CudaSet* b, vector<thrust::device_vector<int_type> >& distinct_tmp)
{

    stack<string> exe_type, exe_value;
    stack<int_type*> exe_vectors, exe_vectors1;
    stack<int_type> exe_nums, exe_nums1;
    string  s1, s2, s1_val, s2_val, grp_type;
    int_type n1, n2, res;
    unsigned int colCount = 0, dist_processed = 0;
    stack<int> col_type;
    stack<string> grp_type1, col_val, exe_value1;
    size_t res_size = 0;
    stack<float_type*> exe_vectors1_d;
    stack<unsigned int> exe_precision, exe_precision1;
    stack<bool> exe_ts;
    bool one_line = 0, ts, free_mem, free_mem1;

    //thrust::device_ptr<bool> d_di(thrust::raw_pointer_cast(a->grp.data()));

    if (a->grp_count && (a->mRecCount != 0))
        res_size = a->grp_count;

    std::clock_t start1 = std::clock();

    for(int i=0; !op_type.empty(); ++i, op_type.pop()) {

        string ss = op_type.front();
        cout << ss << endl;

        if(ss.compare("emit sel_name") != 0) {
            grp_type = "NULL";

            if (ss.compare("COUNT") == 0  || ss.compare("SUM") == 0  || ss.compare("AVG") == 0 || ss.compare("MIN") == 0 || ss.compare("MAX") == 0 || ss.compare("DISTINCT") == 0 || ss.compare("YEAR") == 0 || ss.compare("MONTH") == 0 || ss.compare("DAY") == 0 || ss.compare("CAST") == 0) {

                if(!a->grp_count && ss.compare("YEAR") && ss.compare("MONTH") && ss.compare("DAY") && ss.compare("CAST")) {
                    one_line = 1;
                };

				if (ss.compare("CAST") == 0) {
					exe_type.push(ss);
					exe_value.push(op_value.front());
				}
				else if (ss.compare("YEAR") == 0) {
                    s1_val = exe_value.top();
                    exe_value.pop();
                    exe_type.pop();
                    thrust::device_ptr<int_type> res = thrust::device_malloc<int_type>(a->mRecCount);
                    if(a->ts_cols[s1_val]) {
                        thrust::counting_iterator<unsigned int> begin(0);
                        gpu_getyear ff((const int_type*)thrust::raw_pointer_cast(a->d_columns_int[s1_val].data()),	thrust::raw_pointer_cast(res));
                        thrust::for_each(begin, begin + a->mRecCount, ff);
                        exe_precision.push(0);
						exe_vectors.push(thrust::raw_pointer_cast(res));
						exe_type.push("NAME");
						exe_value.push("");						
                    }
                    else {
						exe_type.push(ss);
						exe_value.push(op_value.front());						
                        exe_precision.push(a->decimal_zeroes[s1_val]);
                    };
                }
                else
                    if (ss.compare("MONTH") == 0) {
                        s1_val = exe_value.top();
                        exe_value.pop();
                        exe_type.pop();
                        thrust::device_ptr<int_type> res = thrust::device_malloc<int_type>(a->mRecCount);
                        thrust::counting_iterator<unsigned int> begin(0);
                        gpu_getmonth ff((const int_type*)thrust::raw_pointer_cast(a->d_columns_int[s1_val].data()),	thrust::raw_pointer_cast(res));
                        thrust::for_each(begin, begin + a->mRecCount, ff);
                        exe_precision.push(0);
                        exe_vectors.push(thrust::raw_pointer_cast(res));
                        exe_type.push("NAME");
                        exe_value.push("");
                    }
                    };

            if (ss.compare("NAME") == 0 || ss.compare("NUMBER") == 0 ) {

                exe_type.push(ss);
                if (ss.compare("NUMBER") == 0) {
                    exe_nums.push(op_nums.front());
                    op_nums.pop();
                    exe_precision.push(op_nums_precision.front());
                    op_nums_precision.pop();
                }
                else
                    if (ss.compare("NAME") == 0) {
                        exe_value.push(op_value.front());
                        ts = a->ts_cols[op_value.front()];
                        op_value.pop();
                    }
            }
            else {
                if (ss.compare("MUL") == 0  || ss.compare("ADD") == 0 || ss.compare("DIV") == 0 || ss.compare("MINUS") == 0) {
                    // get 2 values from the stack
                    s1 = exe_type.top();
                    exe_type.pop();
                    s2 = exe_type.top();
                    exe_type.pop();

                    if (s1.compare("NUMBER") == 0 && s2.compare("NUMBER") == 0) {
                        n1 = exe_nums.top();
                        exe_nums.pop();
                        n2 = exe_nums.top();
                        exe_nums.pop();

                        auto p1 = exe_precision.top();
                        exe_precision.pop();
                        auto p2 = exe_precision.top();
                        exe_precision.pop();
                        auto pres = precision_func(p1, p2, ss);
                        exe_precision.push(pres);
                        if(p1)
                            n1 = n1*(unsigned int)pow(10,p1);
                        if(p2)
                            n2 = n2*(unsigned int)pow(10,p2);

                        if (ss.compare("ADD") == 0 )
                            res = n1+n2;
                        else
                            if (ss.compare("MUL") == 0 )
                                res = n1*n2;
                            else
                                if (ss.compare("DIV") == 0 )
                                    res = n1/n2;
                                else
                                    res = n1-n2;

                        thrust::device_ptr<int_type> p = thrust::device_malloc<int_type>(a->mRecCount);
                        thrust::sequence(p, p+(a->mRecCount),res,(int_type)0);
                        exe_type.push("NAME");
                        exe_value.push("");
                        exe_vectors.push(thrust::raw_pointer_cast(p));
                    }

            }
            }

        }
    };

    if (!a->grp_count) {
        if(!one_line)
            b->mRecCount = a->mRecCount;
        else
            b->mRecCount = 1;
        return one_line;
    }
    else {
        b->mRecCount = res_size;
        return 0;
    };
}


