#include <iostream>
#include "include/einsum_taco/ir/ir.h"
#include "einsum_taco/ir/cleanup.h"
#include "einsum_taco/codegen/codegen_visitor.h"
#include "einsum_taco/codegen/finch_codegen_visitor.h"
#include "einsum_taco/ir/dump_ast.h"
#include "einsum_taco/gstrt/tensor.h"
#include <einsum_taco/parser/heading.h>
#include <string_view>
#include <fstream>
#include <streambuf>
#include <cstring>
#include <unistd.h>
#include <string>


static std::string readFileIntoString(const std::string& path) {
    std::ifstream istrm(path);

    if (!istrm.is_open()) {
        std::cout << "Failed to open file for reading " << path << std::endl;
        std::abort();
    }

    std::stringstream buffer;
    buffer << istrm.rdbuf();

    return buffer.str();
}

static void writeStringToFile(const std::string& filename, const std::string& text) {
    std::ofstream out(filename);
    out << text;
    out.close();
}

inline einsum::Module parse(std::string_view code) {
    char temp_name[17];
    strcpy(temp_name, "dump_test_XXXXXX");
    int f = mkstemp(temp_name);

    if (f == -1) {
        throw std::system_error(errno, std::system_category());
    }
    auto temp = fdopen(f, "r+");
    auto num_written = fwrite(code.data(), 1, code.size(), temp);

    if (num_written != code.size()) {
        throw std::runtime_error("Did not write correct bytes");
    }
    auto status = fseek(temp, 0, SEEK_SET);
    auto module = parse_module(temp);

    fclose(temp);
    close(f);
    return module;
}

int main(int argc, char *argv[]) {
    const char* input_file = argv[1];
    const char* driver_file = argv[2];
    std::string output_dir = argv[3];

    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));

    auto hfile = std::string(cwd) + "/out.h";
    auto cfile = std::string(cwd) + "/out.cc";

    std::fstream oss_cpp(cfile);
    std::fstream oss_h(hfile);
    std::fstream oss_drive(driver_file);

    FinchCodeGenVisitor generator(&oss_cpp, &oss_h, &oss_drive, hfile);

    auto input = readFileIntoString(input_file);

    // parse
    auto mod = std::make_shared<Module>(parse(input));
    auto new_module = apply_default_rewriters(mod);

    // code generation
    new_module->accept(&generator);

    if (argc > 3 && argv[4] == "-c") {
        auto tmp_out = std::string(cwd) + "/out";
        std::stringstream cmdss;
        cmdss << "clang -o " << tmp_out << " -std=c++17 ";
        cmdss << " -I'/usr/local/Cellar/julia/1.8.2/include/julia' -fPIC -I/Users/adadima/mit/commit/Finch.jl/embed  -L'/usr/local/Cellar/julia/1.8.2/lib' -L/Users/adadima/mit/commit/Finch.jl/embed -Wl,-rpath,'/usr/local/Cellar/julia/1.8.2/lib' -ljulia -lfinch";
#if __APPLE__
        cmdss << " -O3";
        cmdss << " -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/";
#endif
        cmdss << " " << cfile << " " << driver_file;
        std::string cmd = cmdss.str();

        FILE* pipe = popen(cmd.c_str(), "r");

        std::array<char, 128> buffer{};
        std::string result;
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            cout << buffer.data();
        }
    }
}
