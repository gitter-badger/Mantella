namespace mant {
  template<typename T = double>
  class ParallelStandardParticleSwarmOptimisation2011 : public PopulationBasedOptimisationAlgorithm<T> {
    public:
      explicit ParallelStandardParticleSwarmOptimisation2011(
          const std::shared_ptr<OptimisationProblem<T>> optimisationProblem,
          const arma::uword populationSize) noexcept;

      void setNeighbourhoodProbability(
          const T neighbourhoodProbability) noexcept;

      void setMaximalAcceleration(
          const T maximalAcceleration) noexcept;

      void setMaximalLocalAttraction(
          const T maximalLocalAttraction) noexcept;

      void setMaximalGlobalAttraction(
          const T maximalGlobalAttraction) noexcept;

      void setMaximalSwarmConvergence(
          const T maximalSwarmConvergence) noexcept;

      void setMaximalCommunicationSteps(
          const arma::uword maximalCommunicationSteps) noexcept;

      bool isRootNode() noexcept;

      std::string toString() const noexcept override;

    protected:
      T neighbourhoodProbability_;
      T maximalAcceleration_;
      T maximalLocalAttraction_;
      T maximalGlobalAttraction_;

      T maximalSwarmConvergence_;

      arma::Mat<T> nodeParticles_;
      arma::Mat<T> nodeVelocities_;
      arma::Col<T> particle_;

      arma::uword neighbourhoodBestParticleIndex_;

      arma::Col<T> attractionCenter_;

      arma::uword maximaCommunicationSteps_;

      arma::Mat<T> nodesBestSolutions_;
      arma::Col<double> localBestSoftConstraintsValues_;
      arma::Col<double> nodesBestObjectiveValues_;

      bool randomizeTopology_;
      arma::Mat<arma::uword> topology_;

      arma::Mat<arma::uword> getRandomNeighbourhoodTopology() noexcept;

      void initialiseSwarm() noexcept;

      void optimiseImplementation() noexcept override;

      T getAcceleration() noexcept;

      arma::Col<T> getVelocity() noexcept;

      bool isConverged() noexcept;
    };

    //
    // Implementation
    //
    template<typename T>
    ParallelStandardParticleSwarmOptimisation2011<T>::ParallelStandardParticleSwarmOptimisation2011(
        const std::shared_ptr<OptimisationProblem<T>> optimisationProblem,
        const arma::uword populationSize) noexcept
      : PopulationBasedOptimisationAlgorithm<T>(optimisationProblem, populationSize),
        nodesBestObjectiveValues_(this->populationSize_ * this->numberOfNodes_),
        nodesBestSolutions_(this->numberOfDimensions_, this->populationSize_ * this->numberOfNodes_),
        randomizeTopology_(true) {
      setNeighbourhoodProbability(std::pow(1.0 - 1.0 / static_cast<T>(this->populationSize_), 3.0));
      setMaximalAcceleration(1.0 / (2.0 * std::log(2.0)));
      setMaximalLocalAttraction(0.5 + std::log(2.0));
      setMaximalGlobalAttraction(maximalLocalAttraction_);
      setMaximalSwarmConvergence(0.05); // TODO Check value within the paper
      setMaximalCommunicationSteps(1);
    }

    template<typename T>
    inline void ParallelStandardParticleSwarmOptimisation2011<T>::optimiseImplementation() noexcept {
      initialiseSwarm();
      randomizeTopology_ = true;
    
      while (!this->isFinished() && !this->isTerminated()) {
        for (std::size_t k = 0; k < maximaCommunicationSteps_; ++k) {
          if (randomizeTopology_) {
              topology_ = getRandomNeighbourhoodTopology();
              randomizeTopology_ = false;
          }

          const arma::Col<arma::uword> permutation = getRandomPermutation(this->populationSize_);

          for (std::size_t n = 0; n < this->populationSize_; ++n) {
            ++this->numberOfIterations_;

            arma::uword particleIndex_ = permutation(n);
            particle_ = nodeParticles_.col(particleIndex_);

            arma::Col<arma::uword> neighbourhoodParticlesIndecies_ = arma::find(topology_.col(particleIndex_));
            static_cast<arma::Col<double>>(nodesBestObjectiveValues_.elem(neighbourhoodParticlesIndecies_)).min(neighbourhoodBestParticleIndex_);
            neighbourhoodBestParticleIndex_ = neighbourhoodParticlesIndecies_(neighbourhoodBestParticleIndex_);

            if (neighbourhoodBestParticleIndex_ == this->nodeRank_ * this->populationSize_ + particleIndex_) {
                attractionCenter_ = (maximalLocalAttraction_ * (nodesBestSolutions_.col(this->nodeRank_ * this->populationSize_ + particleIndex_) - particle_)) / 2.0;
            } else {
                attractionCenter_ = (maximalLocalAttraction_ * (nodesBestSolutions_.col(this->nodeRank_ * this->populationSize_ + particleIndex_) - particle_) + maximalGlobalAttraction_ * (nodesBestSolutions_.col(neighbourhoodBestParticleIndex_) - particle_)) / 3.0;
            }

            arma::Col<T> velocityCandidate = maximalAcceleration_ * getAcceleration() * nodeVelocities_.col(particleIndex_) + getVelocity() * arma::norm(attractionCenter_) + attractionCenter_;
            arma::Col<T> solutionCandidate = particle_ + velocityCandidate;

            const arma::Col<arma::uword> &belowLowerBound = arma::find(solutionCandidate < this->getLowerBounds());
            const arma::Col<arma::uword> &aboveUpperBound = arma::find(solutionCandidate > this->getUpperBounds());

            velocityCandidate.elem(belowLowerBound) *= -0.5;
            velocityCandidate.elem(aboveUpperBound) *= -0.5;

            solutionCandidate.elem(belowLowerBound) = this->getLowerBounds().elem(belowLowerBound);
            solutionCandidate.elem(aboveUpperBound) = this->getUpperBounds().elem(aboveUpperBound);

            nodeVelocities_.col(particleIndex_) = velocityCandidate;
            nodeParticles_.col(particleIndex_) = solutionCandidate;

            const double &objectiveValue = this->getObjectiveValue(solutionCandidate) + this->getSoftConstraintsValue(solutionCandidate);

            if (objectiveValue < nodesBestObjectiveValues_(particleIndex_)) {
                nodesBestObjectiveValues_(this->nodeRank_ * this->populationSize_ + particleIndex_) = objectiveValue;
                nodesBestSolutions_.col(this->nodeRank_ * this->populationSize_ + particleIndex_) = solutionCandidate;
            }

            if (objectiveValue < this->bestObjectiveValue_) {
                this->bestObjectiveValue_ = objectiveValue;
                this->bestParameter_ = solutionCandidate;
            } else {
                randomizeTopology_ = true;
            }

            if (this->isFinished() || this->isTerminated()) {
                break;
            }
          }

          if (this->isFinished() || this->isTerminated()) {
            break;
          }
        }

        arma::Mat<T> localBestSolutionsSend = nodesBestSolutions_.cols(this->nodeRank_ * this->populationSize_, ((this->nodeRank_ + 1) * this->populationSize_) - 1);
        MPI_Allgather(
          localBestSolutionsSend.memptr(),
          this->populationSize_ * this->numberOfDimensions_,
          MPI_DOUBLE,
          nodesBestSolutions_.memptr(),
          this->populationSize_ * this->numberOfDimensions_,
          MPI_DOUBLE,
          MPI_COMM_WORLD);

        arma::Col<double> localBestObjectiveValuesSend = nodesBestObjectiveValues_.subvec(this->nodeRank_ * this->populationSize_, ((this->nodeRank_ + 1) * this->populationSize_) - 1);
        MPI_Allgather(
          localBestObjectiveValuesSend.memptr(),
          this->populationSize_,
          MPI_DOUBLE,
          nodesBestObjectiveValues_.memptr(),
          this->populationSize_,
          MPI_DOUBLE,
          MPI_COMM_WORLD);
		
        arma::uword bestSolutionIndex;
        double bestObjectiveValue = nodesBestObjectiveValues_.min(bestSolutionIndex);
        if (bestObjectiveValue < this->bestObjectiveValue_) {
          this->bestObjectiveValue_ = bestObjectiveValue;
          this->bestParameter_ = nodesBestSolutions_.col(bestSolutionIndex);
        } else {
          randomizeTopology_ = true;
        }

        if (isConverged()) {
          initialiseSwarm();
          randomizeTopology_ = true;
        }
      }
    }

    template<typename T>
    void ParallelStandardParticleSwarmOptimisation2011<T>::setNeighbourhoodProbability(
        const T neighbourhoodProbability) noexcept {
      verify(neighbourhoodProbability >= 0 && neighbourhoodProbability <= 1, "NeighbourhoodProbability must be a value between 0 and 1");
      
      neighbourhoodProbability_ = neighbourhoodProbability;
    }

    template<typename T>
    void ParallelStandardParticleSwarmOptimisation2011<T>::setMaximalAcceleration(
        const T maximalAcceleration) noexcept {
      maximalAcceleration_ = maximalAcceleration;
    }

    template<typename T>
    void ParallelStandardParticleSwarmOptimisation2011<T>::setMaximalLocalAttraction(
        const T maximalLocalAttraction) noexcept {
      maximalLocalAttraction_ = maximalLocalAttraction;
    }

    template<typename T>
    inline void ParallelStandardParticleSwarmOptimisation2011<T>::setMaximalGlobalAttraction(
        const T maximalGlobalAttraction) noexcept {
      maximalGlobalAttraction_ = maximalGlobalAttraction;
    }

    template<typename T>
    void ParallelStandardParticleSwarmOptimisation2011<T>::setMaximalCommunicationSteps(
        const arma::uword maximalCommunicationSteps) noexcept {
      maximaCommunicationSteps_ = maximalCommunicationSteps;
    }

    template<typename T>
    std::string ParallelStandardParticleSwarmOptimisation2011<T>::toString() const noexcept {
        return "ParallelStandardParticleSwarmOptimisation2011";
    }

    template<typename T>
    arma::Mat<arma::uword> ParallelStandardParticleSwarmOptimisation2011<T>::getRandomNeighbourhoodTopology() noexcept {
      arma::Mat<arma::uword> topology = (arma::randu < arma::Mat<T>>(this->populationSize_ * this->numberOfNodes_, this->populationSize_) <= neighbourhoodProbability_);
      topology.diag(-this->nodeRank_ * this->populationSize_) += 1.0;

      return topology;
    }

    template<typename T>
    void ParallelStandardParticleSwarmOptimisation2011<T>::initialiseSwarm() noexcept {
      nodeParticles_ = arma::randu<arma::Mat<T>>(this->numberOfDimensions_, this->populationSize_);
      nodeParticles_.each_col() %= this->getUpperBounds() - this->getLowerBounds();
      nodeParticles_.each_col() += this->getLowerBounds();

      nodeVelocities_ = arma::randu<arma::Mat<T>>(this->numberOfDimensions_, this->populationSize_);
      nodeVelocities_.each_col() %= this->getUpperBounds() - this->getLowerBounds();
      nodeVelocities_.each_col() += this->getLowerBounds();
      nodeVelocities_ -= nodeParticles_;

      nodesBestSolutions_.cols(this->nodeRank_ * this->populationSize_, (this->nodeRank_ + 1) * this->populationSize_ - 1) = nodeParticles_;

      for (std::size_t n = 0; n < this->populationSize_; ++n) {
        ++this->numberOfIterations_;

        arma::Col<T> localBestSolution = nodesBestSolutions_.col(this->nodeRank_ * this->populationSize_ + n);
        double localBestObjectiveValue = this->getObjectiveValue(localBestSolution) + this->getSoftConstraintsValue(localBestSolution);
        nodesBestObjectiveValues_(this->nodeRank_ * this->populationSize_ + n) = localBestObjectiveValue;

        this->updateBestParameter(localBestSolution, 0, localBestObjectiveValue);

        if (this->isFinished() || this->isTerminated()) {
          break;
        }
      }
      

      arma::Mat<T> localBestSolutionsSend = nodesBestSolutions_.cols(this->nodeRank_ * this->populationSize_, ((this->nodeRank_ + 1) * this->populationSize_) - 1);

      MPI_Allgather(
        localBestSolutionsSend.memptr(),
        this->populationSize_ * this->numberOfDimensions_,
        MPI_DOUBLE,
        nodesBestSolutions_.memptr(),
        this->populationSize_ * this->numberOfDimensions_,
        MPI_DOUBLE,
        MPI_COMM_WORLD);

       
      arma::Col<double> localBestObjectiveValuesSend = nodesBestObjectiveValues_.subvec(this->nodeRank_ * this->populationSize_, ((this->nodeRank_ + 1) * this->populationSize_) - 1);

      MPI_Allgather(
        localBestObjectiveValuesSend.memptr(),
        this->populationSize_,
        MPI_DOUBLE,
        nodesBestObjectiveValues_.memptr(),
        this->populationSize_,
        MPI_DOUBLE,
        MPI_COMM_WORLD);
        
      arma::uword bestSolutionIndex;
      this->bestObjectiveValue_ = nodesBestObjectiveValues_.min(bestSolutionIndex);
      this->bestParameter_ = nodesBestSolutions_.col(bestSolutionIndex);
    }

    template<typename T>
    T ParallelStandardParticleSwarmOptimisation2011<T>::getAcceleration() noexcept {
      return std::uniform_real_distribution<T>(0.0, 1.0)(Rng::getGenerator());
    }

    template <typename T>
    arma::Col<T> ParallelStandardParticleSwarmOptimisation2011<T>::getVelocity() noexcept {
      return arma::normalise(arma::randn<arma::Col<T>>(this->numberOfDimensions_)) * std::uniform_real_distribution<T>(0.0, 1.0)(Rng::getGenerator());
    }

    template<typename T>
    void ParallelStandardParticleSwarmOptimisation2011<T>::setMaximalSwarmConvergence(
        const T maximalSwarmConvergence) noexcept {
      maximalSwarmConvergence_ = maximalSwarmConvergence;
    }

    template<typename T>
    bool ParallelStandardParticleSwarmOptimisation2011<T>::isConverged() noexcept {
      return (static_cast<arma::Row<T>>(arma::stddev(nodeParticles_, 1)).max() < maximalSwarmConvergence_);
    }

    template<typename T>
    bool ParallelStandardParticleSwarmOptimisation2011<T>::isRootNode() noexcept{

        if(this->nodeRank_ == 0) {
            return true;
        }
        else {
        	return false;
        }
    }
}
