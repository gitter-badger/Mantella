namespace mant {
  class Rng {
    public:
      Rng() = delete;
      Rng(const Rng&) = delete;
      Rng& operator=(const Rng&) = delete;

      inline static std::mt19937_64& getGenerator() noexcept;

      inline static void setSeed(
          const unsigned int seed) noexcept;

      inline static void setRandomSeed() noexcept;

      inline static unsigned int getSeed() noexcept;

    protected:
      inline static unsigned int& seed_() noexcept;
  };

  //
  // Implementation
  //

  inline std::mt19937_64& Rng::getGenerator() noexcept {
    static std::mt19937_64 generator;
    return generator;
  }

  inline void Rng::setSeed(
      const unsigned int seed) noexcept {
   seed_() = seed;

    getGenerator().seed(seed_());
    arma::arma_rng::set_seed(seed_());

  }

  inline void Rng::setRandomSeed() noexcept {
    arma::arma_rng::set_seed_random();
#if defined(MANTELLA_USE_PARALLEL_ALGORITHMS)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    setSeed(arma::randi<arma::Col<unsigned int>>(1)(0) + rank * arma::randi<arma::Col<unsigned int>>(1)(0));
#else
    setSeed(arma::randi<arma::Col<unsigned int>>(1)(0));
#endif
  }

  inline unsigned int Rng::getSeed() noexcept {
    return seed_();
  }

  inline unsigned int& Rng::seed_() noexcept {
    static unsigned int seed;
    return seed;
  }
}
