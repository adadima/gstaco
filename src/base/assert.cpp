//
// Created by Alexandra Dima on 10/25/21.
//
#include <einsum_taco/base/assert.h>
using std::endl;

namespace einsum {
    einsum::ErrorReport::ErrorReport(const char *file, const char *func, int line, bool condition,
                                     const char *conditionString, einsum::ErrorReport::Kind kind, bool warning)
                                     : file(file), func(func), line(line), condition(condition), conditionString(conditionString),
                                     kind(kind), warning(warning) {
        if (condition) {
            return;
        }
        msg = new std::ostringstream;

        switch (kind) {
            case User:
                if (warning) {
                    (*msg) << "Warning";
                } else {
                    (*msg) << "Error";
                }
                (*msg) << " at " << file << ":" << line << " in " << func << ":" << endl;
                break;
            case Internal:
                (*msg) << "Compiler bug";
                if (warning) {
                    (*msg) << "(warning)";
                }
                (*msg) << " at " << file << ":" << line << " in " << func;
                (*msg) << endl << "Please report it to developers";

                if (conditionString) {
                    (*msg)  << endl << " Condition failed: " << conditionString;
                }
                (*msg) << endl;
                break;
            case Temporary:
                (*msg) << "Temporary assumption broken";
                (*msg) << " at " << file << ":" << line << endl;
                (*msg) << " Not supported yet, but planned for the future";
                if (conditionString) {
                    (*msg) << endl << " Condition failed: " << conditionString;
                }
                (*msg) << endl;
                break;
        }
        (*msg) << " ";
    }

    void einsum::ErrorReport::explodeWithException() {
        EinsumException e (msg->str());
        delete msg;
        throw e;
    }
}