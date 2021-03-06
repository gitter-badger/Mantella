#include "mantella_bits/optimisationProblem/roboticsOptimisationProblem/robotModel/parallelKinematicMachine3PUPS.hpp"

// C++ standard library
#include <cassert>
#include <cmath>

// Mantella
#include "mantella_bits/helper/geometry.hpp"

namespace mant {
  namespace robotics {
    ParallelKinematicMachine3PUPS::ParallelKinematicMachine3PUPS()
        : ParallelKinematicMachine3PUPS(
              // endEffectorJointPositions
              {-0.025561381023353, 0.086293776138137, 0.12, 0.087513292835791, -0.021010082747031, 0.12, -0.061951911812438, -0.065283693391106, 0.12},
              // minimalActiveJointsActuation
              {0.39, 0.39, 0.39},
              // maximalActiveJointsActuation
              {0.95, 0.95, 0.95},
              // redundantJointStartPositions
              {-0.463708870031622, 0.417029254828353, -0.346410161513775, 0.593012363818459, 0.193069033993384, -0.346410161513775, -0.129303493786837, -0.610098288821738, -0.346410161513775},
              // redundantJointEndPositions
              {-0.247202519085512, 0.292029254828353, 0.086602540378444, 0.376506012872349, 0.068069033993384, 0.086602540378444, -0.129303493786837, -0.360098288821738, 0.086602540378444}) {
    }

    ParallelKinematicMachine3PUPS::ParallelKinematicMachine3PUPS(
        const ParallelKinematicMachine3PUPS& parallelKinematicMachine3PUPS)
        : ParallelKinematicMachine3PUPS(parallelKinematicMachine3PUPS.endEffectorJointPositions_, parallelKinematicMachine3PUPS.minimalActiveJointsActuation_, parallelKinematicMachine3PUPS.maximalActiveJointsActuation_, parallelKinematicMachine3PUPS.redundantJointStartPositions_, parallelKinematicMachine3PUPS.redundantJointEndPositions_) {
    }

    ParallelKinematicMachine3PUPS::ParallelKinematicMachine3PUPS(
        const arma::Mat<double>::fixed<3, 3>& endEffectorJointPositions,
        const arma::Row<double>::fixed<3>& minimalActiveJointsActuation,
        const arma::Row<double>::fixed<3>& maximalActiveJointsActuation,
        const arma::Mat<double>::fixed<3, 3>& redundantJointStartPositions,
        const arma::Mat<double>::fixed<3, 3>& redundantJointEndPositions)
        : RobotModel(3, static_cast<arma::Col<double>>(arma::nonzeros(redundantJointEndPositions - redundantJointStartPositions)).n_elem),
          endEffectorJointPositions_(endEffectorJointPositions),
          minimalActiveJointsActuation_(minimalActiveJointsActuation),
          maximalActiveJointsActuation_(maximalActiveJointsActuation),
          redundantJointStartPositions_(redundantJointStartPositions),
          redundantJointEndPositions_(redundantJointEndPositions),
          redundantJointStartToEndPositions_(redundantJointEndPositions_ - redundantJointStartPositions_),
          redundantJointIndicies_(arma::find(arma::any(redundantJointStartToEndPositions_))),
          redundantJointRotationAngles_(3, redundantJointIndicies_.n_elem) {
      for (arma::uword n = 0; n < redundantJointIndicies_.n_elem; ++n) {
        const double redundantJointXAngle = std::atan2(redundantJointStartToEndPositions_(1, n), redundantJointStartToEndPositions_(0, n));
        const double redundantJointYAngle = std::atan2(redundantJointStartToEndPositions_(2, n), redundantJointStartToEndPositions_(1, n));
        redundantJointRotationAngles_.col(n) = arma::Col<double>({std::cos(redundantJointXAngle) * std::cos(redundantJointYAngle), std::sin(redundantJointXAngle) * std::cos(redundantJointYAngle), std::sin(redundantJointYAngle)});
      }
    }

    arma::Cube<double> ParallelKinematicMachine3PUPS::getModelImplementation(
        const arma::Col<double>& endEffectorPose,
        const arma::Row<double>& redundantJointsActuation) const {
      assert(redundantJointsActuation.n_elem == numberOfRedundantJoints_);
      assert(arma::all(redundantJointsActuation >= 0) && arma::all(redundantJointsActuation <= 1));

      arma::Cube<double>::fixed<3, 3, 2> model;

      const arma::Col<double>::fixed<3>& endEffectorPosition = endEffectorPose.subvec(0, 2);
      const double endEffectorRollAngle = endEffectorPose(3);
      const double endEffectorPitchAngle = endEffectorPose(4);
      const double endEffectorYawAngle = endEffectorPose(5);

      model.slice(0) = redundantJointStartPositions_;
      for (arma::uword n = 0; n < redundantJointIndicies_.n_elem; ++n) {
        const arma::uword redundantJointIndex = redundantJointIndicies_(n);
        model.slice(0).col(redundantJointIndex) += redundantJointsActuation(redundantJointIndex) * redundantJointStartToEndPositions_.col(redundantJointIndex);
      }

      model.slice(1) = get3DRotation(endEffectorRollAngle, endEffectorPitchAngle, endEffectorYawAngle) * endEffectorJointPositions_;
      model.slice(1).each_col() += endEffectorPosition;

      return model;
    }

    arma::Row<double> ParallelKinematicMachine3PUPS::getActuationImplementation(
        const arma::Col<double>& endEffectorPose,
        const arma::Row<double>& redundantJointsActuation) const {
      assert(redundantJointsActuation.n_elem == numberOfRedundantJoints_);
      assert(arma::all(redundantJointsActuation >= 0) && arma::all(redundantJointsActuation <= 1));

      const arma::Cube<double>::fixed<3, 3, 2>& model = getModel(endEffectorPose, redundantJointsActuation);

      const arma::Mat<double>::fixed<3, 3>& baseJoints = model.slice(0);
      const arma::Mat<double>::fixed<3, 3>& endEffectorJoints = model.slice(1);

      return arma::sqrt(arma::sum(arma::square(endEffectorJoints - baseJoints)));
    }

    double ParallelKinematicMachine3PUPS::getEndEffectorPoseErrorImplementation(
        const arma::Col<double>& endEffectorPose,
        const arma::Row<double>& redundantJointsActuation) const {
      assert(redundantJointsActuation.n_elem == numberOfRedundantJoints_);
      assert(arma::all(redundantJointsActuation >= 0) && arma::all(redundantJointsActuation <= 1));

      const arma::Cube<double>::fixed<3, 3, 2>& model = getModel(endEffectorPose, redundantJointsActuation);

      const arma::Mat<double>::fixed<3, 3>& baseJoints = model.slice(0);

      const arma::Mat<double>::fixed<3, 3>& endEffectorJoints = model.slice(1);
      arma::Mat<double>::fixed<3, 3> endEffectorJointsRotated = endEffectorJoints;
      endEffectorJointsRotated.each_col() -= endEffectorPose.subvec(0, 1);

      const arma::Mat<double>::fixed<3, 3>& baseToEndEffectorJointPositions = endEffectorJoints - baseJoints;
      const arma::Row<double>::fixed<3>& baseToEndEffectorJointsActuation = arma::sqrt(arma::sum(arma::square(baseToEndEffectorJointPositions)));

      if (arma::any(baseToEndEffectorJointsActuation < minimalActiveJointsActuation_) || arma::any(baseToEndEffectorJointsActuation > maximalActiveJointsActuation_)) {
        return 0.0;
      }

      arma::Mat<double>::fixed<3, 3> forwardKinematic;
      forwardKinematic.head_rows(3) = baseToEndEffectorJointPositions;
      for (arma::uword n = 0; n < baseToEndEffectorJointPositions.n_cols; ++n) {
        forwardKinematic.submat(3, n, 5, n) = arma::cross(endEffectorJointsRotated.col(n), baseToEndEffectorJointPositions.col(n));
      }

      arma::Mat<double> inverseKinematic(6, 3 + redundantJointIndicies_.n_elem, arma::fill::zeros);
      inverseKinematic.diag() = -arma::sqrt(arma::sum(arma::square(baseToEndEffectorJointPositions)));
      for (arma::uword n = 0; n < redundantJointIndicies_.n_elem; ++n) {
        const arma::uword redundantJointIndex = redundantJointIndicies_(n);
        inverseKinematic(n, 3 + n) = arma::dot(baseToEndEffectorJointPositions.col(redundantJointIndex), redundantJointRotationAngles_.col(redundantJointIndex));
      }

      return -1.0 / arma::cond(arma::solve(forwardKinematic.t(), inverseKinematic));
    }

    std::string ParallelKinematicMachine3PUPS::toString() const {
      return "robotics_parallel_kinematic_machine_3pups";
    }
  }
}
