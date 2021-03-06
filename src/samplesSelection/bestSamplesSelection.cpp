#include "mantella_bits/samplesSelection/bestSamplesSelection.hpp"

// C++ standard library
#include <iterator>
#include <unordered_map>
#include <utility>

// Armadillo
#include <armadillo>

namespace mant {
  void BestSamplesSelection::selectImplementation() {
    arma::Row<double> objectiveValues(samples_.size());
    arma::uword n = 0;
    for (const auto& sample : samples_) {
      objectiveValues(++n) = sample.second;
    }

    for (const auto& i : static_cast<arma::Row<arma::uword>>(static_cast<arma::Row<arma::uword>>(arma::sort_index(objectiveValues)).head(numberOfSelectedSamples_))) {
      const auto& selectedSample = std::next(std::begin(samples_), static_cast<decltype(samples_)::difference_type>(i));
      selectedSamples_.insert({selectedSample->first, selectedSample->second});
    }
  }

  std::string BestSamplesSelection::toString() const {
    return "best_samples_selection";
  }
}
