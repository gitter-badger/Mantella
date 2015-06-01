namespace mant {
  namespace bbob {
    template <typename T = double, typename U = double>
    class EllipsoidalFunction : public BlackBoxOptimisationBenchmark<T, U> {
      static_assert(std::is_floating_point<T>::value, "The parameter type T must be a floating point type.");
      static_assert(std::is_floating_point<U>::value, "The codomain type U must be a floating point type.");
    
      public:
        explicit EllipsoidalFunction(
            const std::size_t numberOfDimensions) noexcept;

        std::string toString() const noexcept override;

      protected:
        const arma::Col<T> parameterConditioning_;

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
    EllipsoidalFunction<T, U>::EllipsoidalFunction(
        const std::size_t numberOfDimensions) noexcept
      : BlackBoxOptimisationBenchmark<T, U>(numberOfDimensions),
        parameterConditioning_(this->getParameterConditioning(static_cast<T>(1000000.0L))) {
      this->setParameterTranslation(this->getRandomParameterTranslation());
    }

    template <typename T, typename U>
    T EllipsoidalFunction<T, U>::getObjectiveValueImplementation(
        const arma::Col<T>& parameter) const noexcept {
      return arma::dot(parameterConditioning_, arma::square(this->getOscillatedParameter(parameter)));
    }

    template <typename T, typename U>
    std::string EllipsoidalFunction<T, U>::toString() const noexcept {
      return "bbob_ellipsoidal_function";
    }
    
#if defined(MANTELLA_USE_PARALLEL_ALGORITHMS)
    template <typename T, typename U>
    std::vector<double> EllipsoidalFunction<T, U>::serialise() const noexcept {
      return BlackBoxOptimisationBenchmark<T, T>::serialise();
    }

    template <typename T, typename U>
    void EllipsoidalFunction<T, U>::deserialise(
        const std::vector<double>& serialisedOptimisationProblem) {
      BlackBoxOptimisationBenchmark<T, T>::deserialise(serialisedOptimisationProblem);
    }
#endif
  }
}
