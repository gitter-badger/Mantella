namespace mant {
  namespace bbob {
    template <typename T = double, typename U = double>
    class GallaghersGaussian21hiPeaksFunction : public BlackBoxOptimisationBenchmark<T, U> {
      static_assert(std::is_floating_point<T>::value, "The parameter type T must be a floating point type.");
      static_assert(std::is_floating_point<U>::value, "The codomain type U must be a floating point type.");
    
      public:
        explicit GallaghersGaussian21hiPeaksFunction(
            const std::size_t numberOfDimensions) noexcept;

        virtual void setRotationQ(
            const arma::Mat<T>& rotationQ);

        virtual void setLocalParameterConditionings(
            const arma::Mat<T>& localParameterConditionings);

        virtual void setLocalParameterTranslations(
            const arma::Mat<T>& localParameterTranslations);

        std::string toString() const noexcept override;

      protected:
        const typename arma::Col<T>::template fixed<21> weight_;

        arma::Mat<T> rotationQ_;
        arma::Mat<T> localParameterConditionings_;
        arma::Mat<T> localParameterTranslations_;

        arma::Mat<T> getRandomLocalParameterConditionings() const noexcept;

        arma::Mat<T> getRandomLocalParameterTranslations() const noexcept;

        T getSoftConstraintsValueImplementation(
            const arma::Col<T>& parameter) const noexcept override;

        T getObjectiveValueImplementation(
            const arma::Col<T>& parameter) const noexcept override;
        
#if defined(MANTELLA_USE_PARALLEL_ALGORITHMS)
        friend class OptimisationAlgorithm;
        
        std::vector<double> serialise() const noexcept;

        void deserialise(
            const std::vector<double>& serialisedOptimisationProblem);
#endif
    };

    //
    // Implementation
    //

    template <typename T, typename U>
    GallaghersGaussian21hiPeaksFunction<T, U>::GallaghersGaussian21hiPeaksFunction(
        const std::size_t numberOfDimensions) noexcept
      : BlackBoxOptimisationBenchmark<T, U>(numberOfDimensions),
        weight_(arma::join_cols(arma::Col<T>({static_cast<T>(10.0L)}), arma::linspace<arma::Col<T>>(static_cast<T>(1.1L), static_cast<T>(9.1L), 20))) {
      setRotationQ(getRandomRotationMatrix(this->numberOfDimensions_));
      setLocalParameterConditionings(getRandomLocalParameterConditionings());
      setLocalParameterTranslations(getRandomLocalParameterTranslations());
    }

    template <typename T, typename U>
    void GallaghersGaussian21hiPeaksFunction<T, U>::setRotationQ(
        const arma::Mat<T>& rotationQ) {
      verify(rotationQ.n_rows == this->numberOfDimensions_, "The number of rows must be equal to the number of dimensions");
      verify(isRotationMatrix(rotationQ), "The parameter must be a rotation matrix.");

      rotationQ_ = rotationQ;
    }

    template <typename T, typename U>
    void GallaghersGaussian21hiPeaksFunction<T, U>::setLocalParameterConditionings(
        const arma::Mat<T>& localParameterConditionings) {
      verify(localParameterConditionings.n_rows == this->numberOfDimensions_, "The number of rows must be equal to the number of dimensions");
      verify(localParameterConditionings.n_cols == 21, "The number of columns must be equal to the number of peaks (21).");

      localParameterConditionings_ = localParameterConditionings;
    }

    template <typename T, typename U>
    void GallaghersGaussian21hiPeaksFunction<T, U>::setLocalParameterTranslations(
        const arma::Mat<T>& localParameterTranslations) {
      verify(localParameterTranslations.n_rows == this->numberOfDimensions_, "The number of rows must be equal to the number of dimensions");
      verify(localParameterTranslations.n_cols == 21, "The number of columns must be equal to the number of peaks (21).");

      localParameterTranslations_ = localParameterTranslations;
    }

    template <typename T, typename U>
    arma::Mat<T> GallaghersGaussian21hiPeaksFunction<T, U>::getRandomLocalParameterConditionings() const noexcept {
      arma::Col<T> conditions(21);
      conditions(0) = static_cast<T>(19.0L);
      conditions.tail(conditions.n_elem - 1) = arma::conv_to<arma::Col<T>>::from(getRandomPermutation(conditions.n_elem - 1));

      arma::Mat<T> localParameterConditionings(this->numberOfDimensions_, conditions.n_elem);
      for (std::size_t n = 0; n < conditions.n_elem; ++n) {
        const T& condition = std::pow(static_cast<T>(1000.0L), conditions(n) / static_cast<T>(19.0L));
        localParameterConditionings.col(n) = this->getParameterConditioning(condition) / std::sqrt(condition);
      }

      return localParameterConditionings;
    }

    template <typename T, typename U>
    arma::Mat<T> GallaghersGaussian21hiPeaksFunction<T, U>::getRandomLocalParameterTranslations() const noexcept {
      arma::Mat<T> localParameterTranslations = arma::randu<arma::Mat<T>>(this->numberOfDimensions_, 21) * static_cast<T>(9.8L) - static_cast<T>(4.9L);
      localParameterTranslations.col(0) = static_cast<T>(0.8L) * localParameterTranslations.col(0);

      return localParameterTranslations;
    }

    template <typename T, typename U>
    T GallaghersGaussian21hiPeaksFunction<T, U>::getSoftConstraintsValueImplementation(
        const arma::Col<T>& parameter) const noexcept {
      return this->getBoundConstraintsValue(parameter);
    }

    template <typename T, typename U>
    T GallaghersGaussian21hiPeaksFunction<T, U>::getObjectiveValueImplementation(
        const arma::Col<T>& parameter) const noexcept {
      T maximalValue = std::numeric_limits<U>::lowest();
      for (std::size_t k = 0; k < 21; ++k) {
        const arma::Col<T>& locallyTranslatedParameter = parameter - localParameterTranslations_.col(k);
        maximalValue = std::max(maximalValue, weight_(k) * std::exp(static_cast<T>(-0.5L) / static_cast<T>(this->numberOfDimensions_) * arma::dot(locallyTranslatedParameter, rotationQ_.t() * arma::diagmat(localParameterConditionings_.col(k)) * rotationQ_ * locallyTranslatedParameter)));
      }

      return std::pow(this->getOscillatedValue(static_cast<T>(10.0L) - maximalValue), static_cast<T>(2.0L));
    }

    template <typename T, typename U>
    std::string GallaghersGaussian21hiPeaksFunction<T, U>::toString() const noexcept {
      return "bbob_gallaghers_gaussian_21hi_peaks_function";
    }

#if defined(MANTELLA_USE_PARALLEL_ALGORITHMS)
    template <typename T, typename U>
    std::vector<double> GallaghersGaussian21hiPeaksFunction<T, U>::serialise() const noexcept {
      std::vector<double> serialisedOptimisationProblem = BlackBoxOptimisationBenchmark<T, T>::serialise();
      
      for(std::size_t n = 0; n < rotationQ_.n_elem; ++n) {
        serialisedOptimisationProblem.push_back(rotationQ_(n));
      }
      
      for(std::size_t n = 0; n < localParameterConditionings_.n_elem; ++n) {
        serialisedOptimisationProblem.push_back(localParameterConditionings_(n));
      }
      
      for(std::size_t n = 0; n < localParameterTranslations_.n_elem; ++n) {
        serialisedOptimisationProblem.push_back(localParameterTranslations_(n));
      }
      
      return serialisedOptimisationProblem;
    }

    template <typename T, typename U>
    void GallaghersGaussian21hiPeaksFunction<T, U>::deserialise(
        const std::vector<double>& serialisedOptimisationProblem) {
      rotationQ_.set_size(this->numberOfDimensions_, this->numberOfDimensions_);
      for(std::size_t n = 0; n < rotationQ_.n_elem; ++n) {
        rotationQ_(n) = serialisedOptimisationProblem.pop_back();
      }
      
      localParameterConditionings_.set_size(this->numberOfDimensions_, 21);
      for(std::size_t n = 0; n < localParameterConditionings_.n_elem; ++n) {
        localParameterConditionings_(n) = serialisedOptimisationProblem.pop_back();
      }
      
      localParameterTranslations_.set_size(this->numberOfDimensions_, 21);
      for(std::size_t n = 0; n < localParameterTranslations_.n_elem; ++n) {
        localParameterTranslations_(n) = serialisedOptimisationProblem.pop_back();
      }
        
      BlackBoxOptimisationBenchmark<T, T>::deserialise(serialisedOptimisationProblem);
    }
#endif
  }
}
