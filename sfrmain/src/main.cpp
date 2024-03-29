#include <iostream>
#include <sfr/general.h>

int main() {
    for (auto i : sfr::tukey(10, 5)) {
        std::cout << i << " ";
    }
}