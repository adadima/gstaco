#ifndef EINSUM_ERROR_H
#define EINSUM_ERROR_H

#include <string>
#include <sstream>
#include <ostream>

namespace einsum {

    class EinsumException : public std::runtime_error {
        using std::runtime_error :: runtime_error;
    };

/// Error report (based on Taco's Error.h)
    struct ErrorReport {
        enum Kind { User, Internal, Temporary };

        std::ostringstream *msg;
        const char *file;
        const char *func;
        int line;

        bool condition;
        const char *conditionString;

        Kind kind;
        bool warning;

        ErrorReport(const char *file, const char *func, int line, bool condition,
                    const char *conditionString, Kind kind, bool warning);

        template<typename T>
        ErrorReport &operator<<(T x) {
            if (condition) {
                return *this;
            }
            (*msg) << x;
            return *this;
        }

        ErrorReport &operator<<(std::ostream& (*manip)(std::ostream&)) {
            if (condition) {
                return *this;
            }
            (*msg) << manip;
            return *this;
        }

        ~ErrorReport() noexcept(false) {
            if (condition) {
                return;
            }
            explodeWithException();
        }

        void explodeWithException();
    };

// internal asserts
#ifdef EINSUM_ASSERTS
    #define einsum_iassert(c)                                                     \
    einsum::ErrorReport(__FILE__, __FUNCTION__, __LINE__, (c), #c,              \
                      einsum::ErrorReport::Internal, false)
    #define einsum_ierror                                                         \
    einsum::ErrorReport(__FILE__, __FUNCTION__, __LINE__, false, NULL,          \
                      einsum::ErrorReport::Internal, false)
#else
    struct Dummy {
        template<typename T>
        Dummy &operator<<(T x) {
            return *this;
        }
        // Support for manipulators, such as std::endl
        Dummy &operator<<(std::ostream& (*manip)(std::ostream&)) {
            return *this;
        }
    };

#define einsum_iassert(c) einsum::Dummy()
#define einsum_ierror einsum::Dummy()
#endif

#define einsum_unreachable                                                       \
  einsum_ierror << "reached unreachable location"
}

#endif
