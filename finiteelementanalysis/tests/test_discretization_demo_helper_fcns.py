from finiteelementanalysis import discretization_demo_helper_fcns as di_demo
import numpy as np
from pathlib import Path
import pytest


def node_coords_natural_all_elements(quad_only: bool = False):
    if quad_only:
        node_coords = {
            "D2_nn6_tri": np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0]]),
            "D2_nn8_quad": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
        }
    else:
        node_coords = {
            "D2_nn3_tri": np.array([[1, 0], [0, 1], [0, 0]]),
            "D2_nn6_tri": np.array([[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0]]),
            "D2_nn4_quad": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]),
            "D2_nn8_quad": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
        }
    return node_coords


def linear_fcn_2D(x, y):
    """Defines a simple linear function for interpolation testing."""
    return 2.0 * x - 1.5 * y + 7.0


def test_interpolate_field_natural_coords_single_element_linear():
    """
    Comprehensive test for interpolate_field_natural_coords_single_element function.
    Ensures correct interpolation for various element types using a prescribed function.
    """
    node_coords = node_coords_natural_all_elements()

    for ele_type, nodes in node_coords.items():
        # Evaluate nodal values from the linear function
        node_values = np.array([linear_fcn_2D(x, y) for x, y in nodes]).reshape((-1, 1))

        # Define interpolation points
        if "tri" in ele_type:
            xi_vals = np.array([0.5])
            eta_vals = np.array([0.2])
        else:
            xi_vals = np.array([0.0])
            eta_vals = np.array([0.0])

        # Compute expected interpolation directly from the function
        expected = np.array([[linear_fcn_2D(xi_vals[0], eta_vals[0])]])

        interpolated_vals = di_demo.interpolate_field_natural_coords_single_element(
            ele_type, node_values, xi_vals, eta_vals
        )

        # Ensure the output shape is correct
        assert interpolated_vals.shape == expected.shape, (
            f"{ele_type}: Expected shape {expected.shape}, got {interpolated_vals.shape}"
        )

        # Ensure the interpolated values match the expected results
        assert np.allclose(interpolated_vals, expected, atol=1e-2), (
            f"{ele_type}: Expected {expected}, got {interpolated_vals}"
        )

    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError, match="Unsupported element type"):
        di_demo.interpolate_field_natural_coords_single_element("invalid_type", np.array([1.0]), np.array([0.0]), np.array([0.0]))


def quadratic_fcn_2D(x, y):
    """Defines a quadratic function for interpolation testing."""
    return 2.0 * x**2 - 1.5 * y**2 + 7.0


def test_interpolate_field_natural_coords_single_element_quadratic():
    """
    Test interpolation of a quadratic function using quadratic shape functions.
    Ensures correct quadratic interpolation for higher-order elements.
    """
    node_coords = node_coords_natural_all_elements(True)

    for ele_type, nodes in node_coords.items():
        # Evaluate nodal values from the quadratic function
        node_values = np.array([quadratic_fcn_2D(x, y) for x, y in nodes]).reshape((-1, 1))

        # Define interpolation points
        if "tri" in ele_type:
            xi_vals = np.array([0.5])
            eta_vals = np.array([0.2])
        else:
            xi_vals = np.array([0.0])
            eta_vals = np.array([0.0])

        # Compute expected interpolation directly from the function
        expected = np.array([[quadratic_fcn_2D(xi_vals[0], eta_vals[0])]])

        interpolated_vals = di_demo.interpolate_field_natural_coords_single_element(
            ele_type, node_values, xi_vals, eta_vals
        )

        # Ensure the output shape is correct
        assert interpolated_vals.shape == expected.shape, (
            f"{ele_type}: Expected shape {expected.shape}, got {interpolated_vals.shape}"
        )

        # Ensure the interpolated values match the expected results
        assert np.allclose(interpolated_vals, expected, atol=1e-2), (
            f"{ele_type}: Expected {expected}, got {interpolated_vals}"
        )

    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError, match="Unsupported element type"):
        di_demo.interpolate_field_natural_coords_single_element("invalid_type", np.array([1.0]), np.array([0.0]), np.array([0.0]))


def test_plot_interpolate_field_natural_coords_single_element():
    """
    Test function to verify that `plot_interpolate_field_natural_coords_single_element` runs correctly
    and generates a valid plot file for different element types.
    """
    node_coords = node_coords_natural_all_elements()

    # Create a directory for saving test files
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)

    for ele_type, nodes in node_coords.items():
        # Define a file name inside the test directory
        fname = test_dir / f"test_plot_{ele_type}.png"

        # Delete the file if it already exists
        if fname.exists():
            fname.unlink()

        # Assign arbitrary field values at the nodes
        node_values = np.linspace(1, len(nodes), len(nodes))

        # Call the plotting function
        di_demo.plot_interpolate_field_natural_coords_single_element(str(fname), ele_type, node_values)

        # Ensure the plot file is generated
        assert fname.exists(), f"Plot file {fname} was not created."


def quadratic_fcn_2D_2(x, y):
    """Defines a quadratic function for interpolation testing."""
    return x**2 - 0.5 * x * y + y**2


def test_visualize_isoparametric_mapping_single_element():
    """
    Test function to verify that `visualize_isoparametric_mapping_single_element` runs correctly
    and generates a valid plot file for different element types, ensuring the mapping is clear
    and the function being interpolated is quadratic.
    """
    node_coords = {
        "D2_nn3_tri": np.array([[2, 1], [3, 3], [1, 2]]),
        "D2_nn6_tri": np.array([[2, 1], [3, 3], [1, 2], [2.75, 2], [2, 2.25], [1.5, 1.75]]),
        "D2_nn4_quad": np.array([[1, 1], [4, 0], [3, 3], [0, 2]]),  # Skewed quadrilateral
        "D2_nn8_quad": np.array([[0.5, 0.5], [4, 0], [3, 3], [0, 2], [2.75, 0.5], [3.25, 1.5], [1.25, 2.25], [0.25, 1.25]])  # Skewed with midpoints
    }
    
    # Create a directory for saving test files
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    for ele_type, nodes in node_coords.items():
        # Define a file name inside the test directory
        fname = test_dir / f"test_isoparametric_mapping_{ele_type}.png"
        
        # Delete the file if it already exists
        if fname.exists():
            fname.unlink()
        
        # Evaluate nodal values using the quadratic function
        node_values = np.array([quadratic_fcn_2D_2(x, y) for x, y in nodes])
        
        # Call the visualization function
        di_demo.visualize_isoparametric_mapping_single_element(str(fname), ele_type, nodes, node_values)
        
        # Ensure the plot file is generated
        assert fname.exists(), f"Plot file {fname} was not created."


def test_jacobian_determinant_area():
    """
    Test whether the determinant of the Jacobian matrix correctly captures changes in element area
    for all element types.
    """
    element_types = node_coords_natural_all_elements()

    scale_factor = 2  # Scaling factor for element size

    for ele_type, reference_coords in element_types.items():
        # Scale the element by a factor of 2
        scaled_coords = reference_coords * scale_factor

        # Compute Jacobian determinants at element center
        xi, eta = (0.0, 0.0) if "quad" in ele_type else (1/3, 1/3)
        J_ref = di_demo.compute_jacobian(ele_type, reference_coords, xi, eta)
        J_scaled = di_demo.compute_jacobian(ele_type, scaled_coords, xi, eta)

        # Compute determinants
        det_J_ref = np.linalg.det(J_ref)
        det_J_scaled = np.linalg.det(J_scaled)

        # The determinant should scale as the area (factor squared for 2D elements)
        expected_scaling = scale_factor ** 2
        assert np.isclose(det_J_scaled, expected_scaling * det_J_ref, atol=1e-10), (
            f"{ele_type}: Jacobian determinant did not scale correctly. Expected: {expected_scaling * det_J_ref}, Got: {det_J_scaled}"
        )


def linear_fcn_2D_2(x, y):
    """Defines a simple linear function for interpolation testing."""
    return 2.0 * x - 3.0 * y + 7.0


def linear_fcn_2D_2_deriv(x, y):
    """Returns the derivative of the simple linear function linear_fcn_2D_2."""
    return np.array([2.0, -3.0])  # df/dx = 2.0, df/dy = -3.0


def test_interpolate_gradient_natural_coords_single_element_linear():
    """
    Comprehensive test for interpolate_gradient_natural_coords_single_element function.
    Ensures correct interpolation for various element types using a prescribed function.
    """
    node_coords = node_coords_natural_all_elements()

    test_points = {
        "D2_nn3_tri": [(0.5, 0.2)],
        "D2_nn6_tri": [(0.5, 0.2)],
        "D2_nn4_quad": [(0.0, 0.0)],
        "D2_nn8_quad": [(0.0, 0.0)]
    }

    for ele_type, nodes in node_coords.items():
        # Evaluate nodal values from the linear function
        node_values = np.array([linear_fcn_2D_2(x, y) for x, y in nodes]).reshape((-1, 1))

        for xi, eta in test_points[ele_type]:
            # Compute expected interpolation directly from the function
            expected = np.array(linear_fcn_2D_2_deriv(xi, eta)).reshape((2, 1))

            interpolated_vals = di_demo.interpolate_gradient_natural_coords_single_element(
                ele_type, node_values, np.array([xi]), np.array([eta])
            ).reshape((2, 1))

            # Ensure the output shape is correct
            assert interpolated_vals.shape == expected.shape, (
                f"{ele_type}: Expected shape {expected.shape}, got {interpolated_vals.shape}"
            )

            # Ensure the interpolated values match the expected results
            assert np.allclose(interpolated_vals.flatten(), expected.flatten(), atol=1e-2), (
                f"{ele_type}: Expected {expected.flatten()}, got {interpolated_vals.flatten()}"
            )

    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError, match="Unsupported element type"):
        di_demo.interpolate_gradient_natural_coords_single_element("invalid_type", np.array([[1.0]]), np.array([0.0]), np.array([0.0]))


def quadratic_fcn_2D_3(x, y):
    """Defines a quadratic function for interpolation testing."""
    return x**2 - 2 * x * y + y**2 + 5


def quadratic_fcn_2D_3_deriv(x, y):
    """Returns the derivative of the quadratic function quadratic_fcn_2D_2."""
    return np.array([2*x - 2*y, 2*y - 2*x])  # df/dx, df/dy


def test_interpolate_gradient_natural_coords_single_element_quadratic():
    """
    Comprehensive test for interpolate_gradient_natural_coords_single_element function.
    Ensures correct interpolation for quadratic element types using a prescribed quadratic function.
    """
    node_coords = node_coords_natural_all_elements(True)

    test_points = {
        "D2_nn6_tri": [(0.5, 0.2), (1/3, 1/3)],
        "D2_nn8_quad": [(0.0, 0.0), (0.5, 0.5)]
    }

    for ele_type, nodes in node_coords.items():
        # Evaluate nodal values from the quadratic function
        node_values = np.array([quadratic_fcn_2D_3(x, y) for x, y in nodes]).reshape((-1, 1))

        for xi, eta in test_points[ele_type]:
            # Compute expected interpolation directly from the function
            expected = np.array(quadratic_fcn_2D_3_deriv(xi, eta)).reshape((2, 1))

            interpolated_vals = di_demo.interpolate_gradient_natural_coords_single_element(
                ele_type, node_values, np.array([xi]), np.array([eta])
            ).reshape((2, 1))

            # Ensure the output shape is correct
            assert interpolated_vals.shape == expected.shape, (
                f"{ele_type}: Expected shape {expected.shape}, got {interpolated_vals.shape}"
            )

            # Ensure the interpolated values match the expected results
            assert np.allclose(interpolated_vals.flatten(), expected.flatten(), atol=1e-2), (
                f"{ele_type}: Expected {expected.flatten()}, got {interpolated_vals.flatten()}"
            )

    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError, match="Unsupported element type"):
        di_demo.interpolate_gradient_natural_coords_single_element("invalid_type", np.array([[1.0]]), np.array([0.0]), np.array([0.0]))


def linear_vector_field_2D(x, y):
    """Defines a linear vector field for interpolation testing."""
    return np.array([2*x + y, -x + 3*y])  # u(x, y), v(x, y)


def linear_vector_field_2D_deriv(x, y):
    """Returns the Jacobian of the linear vector field."""
    return np.array([
        [2, 1],  # ∂u/∂x, ∂u/∂y
        [-1, 3]  # ∂v/∂x, ∂v/∂y
    ])


def test_interpolate_gradient_natural_coords_single_element_vector_linear():
    """
    Test for interpolate_gradient_natural_coords_single_element function with linear vector fields.
    Ensures correct interpolation for all element types using a prescribed linear vector field.
    """
    node_coords = node_coords_natural_all_elements()

    test_points = {
        "D2_nn3_tri": [(0.5, 0.2)],
        "D2_nn6_tri": [(0.5, 0.2)],
        "D2_nn4_quad": [(0.0, 0.0)],
        "D2_nn8_quad": [(0.0, 0.0)]
    }

    for ele_type, nodes in node_coords.items():
        # Evaluate nodal values from the linear vector field
        node_values = np.array([linear_vector_field_2D(x, y) for x, y in nodes]).reshape((-1, 2))

        for xi, eta in test_points[ele_type]:
            # Compute expected interpolation directly from the function's derivative
            expected = linear_vector_field_2D_deriv(xi, eta)  # Shape (2,2)

            interpolated_vals = di_demo.interpolate_gradient_natural_coords_single_element(
                ele_type, node_values, np.array([xi]), np.array([eta])
            ).reshape((2, 2)).T

            # Ensure the output shape is correct
            assert interpolated_vals.shape == expected.shape, (
                f"{ele_type}: Expected shape {expected.shape}, got {interpolated_vals.shape}"
            )

            # Ensure the interpolated values match the expected results
            assert np.allclose(interpolated_vals, expected, atol=1e-2), (
                f"{ele_type}: Expected {expected.flatten()}, got {interpolated_vals.flatten()}"
            )

    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError, match="Unsupported element type"):
        di_demo.interpolate_gradient_natural_coords_single_element("invalid_type", np.array([[1.0, 1.0]]), np.array([0.0]), np.array([0.0]))


def quadratic_vector_field_2D(x, y):
    """Defines a quadratic vector field for interpolation testing."""
    return np.array([x**2 - y**2, 2*x*y])  # u(x, y), v(x, y)


def quadratic_vector_field_2D_deriv(x, y):
    """Returns the Jacobian of the quadratic vector field."""
    return np.array([
        [2*x, -2*y],  # ∂u/∂x, ∂u/∂y
        [2*y, 2*x]    # ∂v/∂x, ∂v/∂y
    ])


def test_interpolate_gradient_natural_coords_single_element_vector_quadratic():
    """
    Test for interpolate_gradient_natural_coords_single_element function with quadratic vector fields.
    Ensures correct interpolation for quadratic element types using a prescribed quadratic vector field.
    """
    node_coords = node_coords_natural_all_elements(True)
    
    test_points = {
        "D2_nn6_tri": [(0.5, 0.2), (1/3, 1/3)],
        "D2_nn8_quad": [(0.0, 0.0), (0.5, 0.5)]
    }
    
    for ele_type, nodes in node_coords.items():
        # Evaluate nodal values from the quadratic vector field
        node_values = np.array([quadratic_vector_field_2D(x, y) for x, y in nodes]).reshape((-1, 2))
        
        for xi, eta in test_points[ele_type]:
            # Compute expected interpolation directly from the function's derivative
            expected = quadratic_vector_field_2D_deriv(xi, eta)  # Shape (2,2)
            
            interpolated_vals = di_demo.interpolate_gradient_natural_coords_single_element(
                ele_type, node_values, np.array([xi]), np.array([eta])
            ).reshape((2, 2)).T
            
            # Ensure the output shape is correct
            assert interpolated_vals.shape == expected.shape, (
                f"{ele_type}: Expected shape {expected.shape}, got {interpolated_vals.shape}"
            )
            
            # Ensure the interpolated values match the expected results
            assert np.allclose(interpolated_vals, expected, atol=1e-2), (
                f"{ele_type}: Expected {expected.flatten()}, got {interpolated_vals.flatten()}"
            )
    
    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError, match="Unsupported element type"):
        di_demo.interpolate_gradient_natural_coords_single_element("invalid_type", np.array([[1.0, 1.0]]), np.array([0.0]), np.array([0.0]))


def linear_scalar_field_2D(x, y):
    """Defines a linear scalar field for interpolation testing."""
    return 2*x + 3*y + 5  # f(x, y)


def linear_scalar_field_2D_deriv(x, y):
    """Computes the analytical gradient of the linear scalar field in physical coordinates."""
    return np.array([2, 3])  # df/dx, df/dy


def test_transform_gradient_to_physical():
    """
    Test transform_gradient_to_physical function by comparing numerical transformation
    with an analytical solution in physical coordinates.
    """
    node_coords_physical = {
        "D2_nn3_tri": np.array([[1, 1], [3, 1], [2, 3]]),
        "D2_nn6_tri": np.array([[1, 1], [3, 1], [2, 3], [2, 1], [2.5, 2], [1.5, 2]]),
        "D2_nn4_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
        "D2_nn8_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [2, 1], [1, 2], [0, 1]])
    }

    test_points_natural = {
        "D2_nn3_tri": [[1, 0], [0, 1], [0, 0], [1/3, 1/3], [0.5, 0.2]],
        "D2_nn6_tri": [[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0], [1/3, 1/3]],
        "D2_nn4_quad": [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0], [0.5, 0.5]],
        "D2_nn8_quad": [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5]]
    }

    for ele_type, nodes_physical in node_coords_physical.items():
        # Compute function values at element nodes
        node_values = np.array([linear_scalar_field_2D(x, y) for x, y in nodes_physical]).reshape((-1, 1))
        
        for xi_eta in test_points_natural[ele_type]:
            xi, eta = xi_eta
            
            # Map the test point from natural to physical coordinates
            x_mapped = di_demo.interpolate_field_natural_coords_single_element(ele_type, nodes_physical[:, 0], [xi], [eta]).flatten()[0]
            y_mapped = di_demo.interpolate_field_natural_coords_single_element(ele_type, nodes_physical[:, 1], [xi], [eta]).flatten()[0]
            
            # Compute analytical gradient at the mapped physical coordinates
            analytical_grad_physical = linear_scalar_field_2D_deriv(x_mapped, y_mapped).reshape((2, 1))
            
            # Compute the numerical gradient in natural coordinates
            gradient_natural = di_demo.interpolate_gradient_natural_coords_single_element(
                ele_type, node_values, np.array([xi]), np.array([eta])
            )
            
            # Transform the gradient to physical coordinates
            transformed_grad = di_demo.transform_gradient_to_physical(
                ele_type, nodes_physical, np.array([xi]), np.array([eta]), gradient_natural
            ).reshape((2, 1))
            
            # Ensure the output shape is correct
            assert transformed_grad.shape == analytical_grad_physical.shape, (
                f"{ele_type}: Expected shape {analytical_grad_physical.shape}, got {transformed_grad.shape}"
            )
            
            # Ensure the transformed values match the expected results
            assert np.allclose(transformed_grad.flatten(), analytical_grad_physical.flatten(), atol=1e-2), (
                f"{ele_type}: Expected {analytical_grad_physical.flatten()}, got {transformed_grad.flatten()}"
            )


def quadratic_scalar_field_2D(x, y):
    """Defines a quadratic scalar field for interpolation testing."""
    return x**2 - y**2 + 3*x*y + 5  # f(x, y)


def quadratic_scalar_field_2D_deriv(x, y):
    """Computes the analytical gradient of the quadratic scalar field in physical coordinates."""
    return np.array([2*x + 3*y, -2*y + 3*x])  # df/dx, df/dy


def test_transform_gradient_to_physical_quadratic():
    """
    Test transform_gradient_to_physical function by comparing numerical transformation
    with an analytical solution in physical coordinates for quadratic elements only.
    """
    node_coords_physical = {
        "D2_nn6_tri": np.array([[1, 1], [3, 1], [2, 3], [2, 1], [2.5, 2], [1.5, 2]]),
        "D2_nn8_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [2, 1], [1, 2], [0, 1]])
    }
    
    test_points_natural = {
        "D2_nn6_tri": [[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0], [1/3, 1/3]],
        "D2_nn8_quad": [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5]]
    }
    
    for ele_type, nodes_physical in node_coords_physical.items():
        # Compute function values at element nodes
        node_values = np.array([quadratic_scalar_field_2D(x, y) for x, y in nodes_physical]).reshape((-1, 1))
        
        for xi_eta in test_points_natural[ele_type]:
            xi, eta = xi_eta
            
            # Map the test point from natural to physical coordinates
            x_mapped = di_demo.interpolate_field_natural_coords_single_element(ele_type, nodes_physical[:, 0], [xi], [eta]).flatten()[0]
            y_mapped = di_demo.interpolate_field_natural_coords_single_element(ele_type, nodes_physical[:, 1], [xi], [eta]).flatten()[0]
            
            # Compute analytical gradient at the mapped physical coordinates
            analytical_grad_physical = quadratic_scalar_field_2D_deriv(x_mapped, y_mapped).reshape((2, 1))
            
            # Compute the numerical gradient in natural coordinates
            gradient_natural = di_demo.interpolate_gradient_natural_coords_single_element(
                ele_type, node_values, np.array([xi]), np.array([eta])
            )
            
            # Transform the gradient to physical coordinates
            transformed_grad = di_demo.transform_gradient_to_physical(
                ele_type, nodes_physical, np.array([xi]), np.array([eta]), gradient_natural
            ).reshape((2, 1))
            
            # Ensure the output shape is correct
            assert transformed_grad.shape == analytical_grad_physical.shape, (
                f"{ele_type}: Expected shape {analytical_grad_physical.shape}, got {transformed_grad.shape}"
            )
            
            # Ensure the transformed values match the expected results
            assert np.allclose(transformed_grad.flatten(), analytical_grad_physical.flatten(), atol=1e-2), (
                f"{ele_type}: Expected {analytical_grad_physical.flatten()}, got {transformed_grad.flatten()}"
            )


def linear_vector_field_2D_2(x, y):
    """Defines a linear vector field for interpolation testing."""
    return np.array([2*x + y, -x + 3*y])  # u(x, y), v(x, y)


def linear_vector_field_2D_deriv_2(x, y):
    """Computes the analytical gradient (Jacobian) of the linear vector field in physical coordinates."""
    return np.array([
        [2, 1],  # ∂u/∂x, ∂u/∂y
        [-1, 3]  # ∂v/∂x, ∂v/∂y
    ])


def test_transform_gradient_to_physical_linear_vector():
    """
    Test transform_gradient_to_physical function by comparing numerical transformation
    with an analytical solution in physical coordinates for all element types using a linear vector field.
    """
    node_coords_physical = {
        "D2_nn3_tri": np.array([[1, 1], [3, 1], [2, 3]]),
        "D2_nn6_tri": np.array([[1, 1], [3, 1], [2, 3], [2, 1], [2.5, 2], [1.5, 2]]),
        "D2_nn4_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
        "D2_nn8_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [2, 1], [1, 2], [0, 1]])
    }
    
    test_points_natural = {
        "D2_nn3_tri": [[1, 0], [0, 1], [0, 0], [1/3, 1/3]],
        "D2_nn6_tri": [[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0], [1/3, 1/3]],
        "D2_nn4_quad": [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]],
        "D2_nn8_quad": [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]]
    }
    
    for ele_type, nodes_physical in node_coords_physical.items():
        # Compute function values at element nodes
        node_values = np.array([linear_vector_field_2D_2(x, y) for x, y in nodes_physical]).reshape((-1, 2))
        
        for xi_eta in test_points_natural[ele_type]:
            xi, eta = xi_eta
            
            # Map the test point from natural to physical coordinates
            x_mapped = di_demo.interpolate_field_natural_coords_single_element(ele_type, nodes_physical[:, 0], [xi], [eta]).flatten()[0]
            y_mapped = di_demo.interpolate_field_natural_coords_single_element(ele_type, nodes_physical[:, 1], [xi], [eta]).flatten()[0]
            
            # Compute analytical gradient at the mapped physical coordinates
            analytical_grad_physical = linear_vector_field_2D_deriv_2(x_mapped, y_mapped)
            
            # Compute the numerical gradient in natural coordinates
            gradient_natural = di_demo.interpolate_gradient_natural_coords_single_element(
                ele_type, node_values, np.array([xi]), np.array([eta])
            )
            
            # Transform the gradient to physical coordinates
            transformed_grad = di_demo.transform_gradient_to_physical(
                ele_type, nodes_physical, np.array([xi]), np.array([eta]), gradient_natural
            ).reshape((2, 2)).T
            
            # Ensure the output shape is correct
            assert transformed_grad.shape == analytical_grad_physical.shape, (
                f"{ele_type}: Expected shape {analytical_grad_physical.shape}, got {transformed_grad.shape}"
            )
            
            # Ensure the transformed values match the expected results
            assert np.allclose(transformed_grad, analytical_grad_physical, atol=1e-2), (
                f"{ele_type}: Expected {analytical_grad_physical.flatten()}, got {transformed_grad.flatten()}"
            )


def quadratic_vector_field_2D_2(x, y):
    """Defines a quadratic vector field for interpolation testing."""
    return np.array([x**2 - y**2, 2*x*y])  # u(x, y), v(x, y)


def quadratic_vector_field_2D_deriv_2(x, y):
    """Computes the analytical gradient (Jacobian) of the quadratic vector field in physical coordinates."""
    return np.array([
        [2*x, -2*y],  # ∂u/∂x, ∂u/∂y
        [2*y, 2*x]    # ∂v/∂x, ∂v/∂y
    ])


def test_transform_gradient_to_physical_quadratic_vector():
    """
    Test transform_gradient_to_physical function by comparing numerical transformation
    with an analytical solution in physical coordinates for quadratic elements using a quadratic vector field.
    """
    node_coords_physical = {
        "D2_nn6_tri": np.array([[1, 1], [3, 1], [2, 3], [2, 1], [2.5, 2], [1.5, 2]]),
        "D2_nn8_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [2, 1], [1, 2], [0, 1]])
    }
    
    test_points_natural = {
        "D2_nn6_tri": [[1, 0], [0, 1], [0, 0], [0.5, 0.5], [0, 0.5], [0.5, 0], [1/3, 1/3]],
        "D2_nn8_quad": [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5]]
    }
    
    for ele_type, nodes_physical in node_coords_physical.items():
        # Compute function values at element nodes
        node_values = np.array([quadratic_vector_field_2D_2(x, y) for x, y in nodes_physical]).reshape((-1, 2))
        
        for xi_eta in test_points_natural[ele_type]:
            xi, eta = xi_eta
            
            # Map the test point from natural to physical coordinates
            x_mapped = di_demo.interpolate_field_natural_coords_single_element(ele_type, nodes_physical[:, 0], [xi], [eta]).flatten()[0]
            y_mapped = di_demo.interpolate_field_natural_coords_single_element(ele_type, nodes_physical[:, 1], [xi], [eta]).flatten()[0]
            
            # Compute analytical gradient at the mapped physical coordinates
            analytical_grad_physical = quadratic_vector_field_2D_deriv_2(x_mapped, y_mapped)
            
            # Compute the numerical gradient in natural coordinates
            gradient_natural = di_demo.interpolate_gradient_natural_coords_single_element(
                ele_type, node_values, np.array([xi]), np.array([eta])
            )
            
            # Transform the gradient to physical coordinates
            transformed_grad = di_demo.transform_gradient_to_physical(
                ele_type, nodes_physical, np.array([xi]), np.array([eta]), gradient_natural
            ).reshape((2, 2)).T
            
            # Ensure the output shape is correct
            assert transformed_grad.shape == analytical_grad_physical.shape, (
                f"{ele_type}: Expected shape {analytical_grad_physical.shape}, got {transformed_grad.shape}"
            )
            
            # Ensure the transformed values match the expected results
            assert np.allclose(transformed_grad, analytical_grad_physical, atol=1e-2), (
                f"{ele_type}: Expected {analytical_grad_physical.flatten()}, got {transformed_grad.flatten()}"
            )


def test_gauss_pts_and_weights():
    """
    Test gauss_pts_and_weights function to verify correctness of quadrature points and weights
    for different element types and number of integration points.
    """
    element_types = ["D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad"]
    
    valid_num_pts = {
        "D2_nn3_tri": [1, 3, 4],
        "D2_nn6_tri": [1, 3, 4],
        "D2_nn4_quad": [1, 4, 9],
        "D2_nn8_quad": [1, 4, 9]
    }
    
    for ele_type in element_types:
        for num_pts in valid_num_pts[ele_type]:
            gauss_pts, gauss_weights = di_demo.gauss_pts_and_weights(ele_type, num_pts)
            
            # Ensure the output shapes are correct
            assert gauss_pts.shape == (2, num_pts), (
                f"{ele_type}: Expected Gauss points shape (2, {num_pts}), got {gauss_pts.shape}."
            )
            assert gauss_weights.shape == (num_pts, 1), (
                f"{ele_type}: Expected Gauss weights shape ({num_pts}, 1), got {gauss_weights.shape}."
            )
            
            # Ensure weights sum to a reasonable value (checks for correctness in quadrature sum)
            assert np.isclose(np.sum(gauss_weights), 1.0, atol=0.5) or np.isclose(np.sum(gauss_weights), 4.0, atol=0.5), (
                f"{ele_type}: Unexpected sum of Gauss weights: {np.sum(gauss_weights)}."
            )
    
    # Ensure an invalid element type raises an error
    with pytest.raises(ValueError, match="Unsupported element type"):
        di_demo.gauss_pts_and_weights("invalid_type", 3)
    
    # Ensure an invalid number of points raises an error
    with pytest.raises(ValueError, match="num_pts must be 1, 3, or 4"):
        di_demo.gauss_pts_and_weights("D2_nn3_tri", 2)


def test_visualize_gauss_pts():
    """
    Test function to verify that `visualize_gauss_pts` runs correctly and generates
    a valid plot file for different element types and Gauss point configurations.
    """
    element_types = ["D2_nn3_tri", "D2_nn6_tri", "D2_nn4_quad", "D2_nn8_quad"]
    valid_num_pts = {
        "D2_nn3_tri": [1, 3, 4],
        "D2_nn6_tri": [1, 3, 4],
        "D2_nn4_quad": [1, 4, 9],
        "D2_nn8_quad": [1, 4, 9]
    }
    
    # Create a directory for saving test files
    test_dir = Path(__file__).parent / "files"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    for ele_type in element_types:
        for num_pts in valid_num_pts[ele_type]:
            # Define a file name inside the test directory
            fname = test_dir / f"test_visualize_gauss_pts_{ele_type}_{num_pts}.png"
            
            # Delete the file if it already exists
            if fname.exists():
                fname.unlink()
            
            # Call the plotting function
            di_demo.visualize_gauss_pts(str(fname), ele_type, num_pts)
            
            # Ensure the plot file is generated
            assert fname.exists(), f"Plot file {fname} was not created."


def fcn_linear_1(x, y):
    """Defines a simple function for testing integration."""
    return 3.0 * x + 10.0 * y


def deriv_linear_1(x, y):
    """Computes the analytical gradient of fcn(x, y)."""
    return np.asarray([3.0, 10.0])  # df/dx = 3.0, df/dy = 0


def compute_element_area(node_coords):
    """Computes the area of a triangular or quadrilateral element assuming mid-edge nodes are at midpoints."""
    if node_coords.shape[0] == 3 or node_coords.shape[0] == 6:  # Triangular elements
        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]
        return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    elif node_coords.shape[0] == 4 or node_coords.shape[0] == 8:  # Quadrilateral elements
        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]
        x4, y4 = node_coords[3]
        return 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))
    else:
        raise ValueError("Unsupported element type for area computation.")


def integral_of_deriv_1(node_coords):
    """Computes the analytical integral of the derivative using the element's area."""
    area = compute_element_area(node_coords)
    return np.array([3.0 * area, 10.0 * area]).reshape((2, 1))  # Integral of df/dx over the area


def analytical_integral_of_derivative_1(ele_type, node_coords):
    """
    Computes the analytical integral of the derivative of fcn(x, y) over the given element.
    """
    return integral_of_deriv_1(node_coords)


def test_compute_integral_of_derivative_linear_scalar():
    """
    Test compute_integral_of_derivative function by comparing numerical integration results
    with analytical solutions across multiple quadrature rules.
    """
    node_coords_physical = {
        "D2_nn3_tri": np.array([[1, 1], [3, 1], [2, 3]]),
        "D2_nn6_tri": np.array([[1, 1], [3, 1], [2, 3], [2, 1], [2.5, 2], [1.5, 2]]),
        "D2_nn4_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
        "D2_nn8_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [2, 1], [1, 2], [0, 1]])
    }
    
    valid_num_pts = {
        "D2_nn3_tri": [1, 3, 4],
        "D2_nn6_tri": [1, 3, 4],
        "D2_nn4_quad": [1, 4, 9],
        "D2_nn8_quad": [1, 4, 9]
    }
    
    for ele_type, nodes_physical in node_coords_physical.items():
        for num_gauss_pts in valid_num_pts[ele_type]:
            # Compute nodal values from fcn(x, y)
            nodal_values = np.array([[fcn_linear_1(x, y) for x, y in nodes_physical]]).T
            
            # Compute numerical integral
            integral_numerical = di_demo.compute_integral_of_derivative(ele_type, num_gauss_pts, nodes_physical, nodal_values)
            
            # Compute analytical integral
            integral_analytical = analytical_integral_of_derivative_1(ele_type, nodes_physical)
            
            # Ensure the output shape is correct
            assert integral_numerical.shape == integral_analytical.shape, (
                f"{ele_type}, {num_gauss_pts}-point quadrature: Expected shape {integral_analytical.shape}, got {integral_numerical.shape}"
            )
            
            # Ensure the computed integral matches the analytical solution
            assert np.allclose(integral_numerical, integral_analytical, atol=1e-2), (
                f"{ele_type}, {num_gauss_pts}-point quadrature: Expected {integral_analytical.flatten()}, got {integral_numerical.flatten()}"
            )


def fcn_linear_2(x, y):
    """Defines a simple function for testing integration."""
    return np.asarray([[3.0 * x], [20.0 * y]])


def deriv_linear_2(x, y):
    """Computes the analytical gradient of fcn(x, y)."""
    return np.asarray([[3.0, 0], [0, 20.0]])  # df/dx = 3.0, df/dy = 0


def integral_of_deriv_2(node_coords):
    """Computes the analytical integral of the derivative using the element's area."""
    area = compute_element_area(node_coords)
    return np.array([[3.0 * area, 0], [0, 20.0 * area]])


def analytical_integral_of_derivative_2(node_coords):
    """
    Computes the analytical integral of the derivative of fcn(x, y) over the given element.
    """
    return integral_of_deriv_2(node_coords)


def test_compute_integral_of_derivative_linear_vector():
    """
    Test compute_integral_of_derivative function by comparing numerical integration results
    with analytical solutions across multiple quadrature rules.
    """
    node_coords_physical = {
        "D2_nn3_tri": np.array([[1, 1], [3, 1], [2, 3]]),
        "D2_nn6_tri": np.array([[1, 1], [3, 1], [2, 3], [2, 1], [2.5, 2], [1.5, 2]]),
        "D2_nn4_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
        "D2_nn8_quad": np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [2, 1], [1, 2], [0, 1]])
    }
    
    valid_num_pts = {
        "D2_nn3_tri": [1, 3, 4],
        "D2_nn6_tri": [1, 3, 4],
        "D2_nn4_quad": [1, 4, 9],
        "D2_nn8_quad": [1, 4, 9]
    }
    
    for ele_type, nodes_physical in node_coords_physical.items():
        for num_gauss_pts in valid_num_pts[ele_type]:
            # Compute nodal values from fcn(x, y)
            nodal_values = np.array([fcn_linear_2(x, y) for x, y in nodes_physical]).squeeze()
            
            # Compute numerical integral
            integral_numerical = di_demo.compute_integral_of_derivative(ele_type, num_gauss_pts, nodes_physical, nodal_values)
            
            # Compute analytical integral
            integral_analytical = analytical_integral_of_derivative_2(nodes_physical)
            
            # Ensure the output shape is correct
            assert integral_numerical.shape == integral_analytical.shape, (
                f"{ele_type}, {num_gauss_pts}-point quadrature: Expected shape {integral_analytical.shape}, got {integral_numerical.shape}"
            )
            
            # Ensure the computed integral matches the analytical solution
            assert np.allclose(integral_numerical, integral_analytical, atol=1e-2), (
                f"{ele_type}, {num_gauss_pts}-point quadrature: Expected {integral_analytical.flatten()}, got {integral_numerical.flatten()}"
            )

