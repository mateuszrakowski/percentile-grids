# Test Suite for Percentile-Grids Application

This directory contains comprehensive tests for the percentile-grids application, covering all major components and functionality.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── test_data_processing.py     # Tests for data processing module
├── test_brain_structures.py    # Tests for brain structure definitions
├── test_data_cache.py          # Tests for caching utilities
├── test_model_engine.py        # Tests for GAMLSS model engine
├── test_web_interface.py       # Tests for Streamlit web interface
├── test_runner.py              # Test runner script
└── README.md                   # This file
```

## Test Coverage

### 1. Data Processing Tests (`test_data_processing.py`)
- **CSV Input Processing**: Tests for processing raw CSV data into structured format
- **Structure Volume Summation**: Tests for aggregating brain structure volumes
- **File Format Handling**: Tests for CSV and Excel file processing
- **Database Operations**: Tests for SQLite database operations
- **Data Validation**: Tests for data integrity and format validation

### 2. Brain Structures Tests (`test_brain_structures.py`)
- **Structure Definitions**: Tests for all brain structure classes
- **Inheritance Hierarchy**: Tests for proper class inheritance
- **Model Serialization**: Tests for Pydantic model functionality
- **Structure Aggregation**: Tests for volume summation logic

### 3. Data Cache Tests (`test_data_cache.py`)
- **Cache Key Generation**: Tests for MD5-based cache key generation
- **Disk Caching**: Tests for file-based caching functionality
- **Memory Caching**: Tests for in-memory cache optimization
- **Cache Management**: Tests for cache clearing and maintenance
- **Error Handling**: Tests for cache error scenarios

### 4. Model Engine Tests (`test_model_engine.py`)
- **GAMLSS Model**: Tests for GAMLSS model initialization and fitting
- **Model Metrics**: Tests for AIC, BIC, and deviance calculations
- **Percentile Calculation**: Tests for percentile curve generation
- **Model Serialization**: Tests for saving and loading models
- **Plotting**: Tests for matplotlib figure generation
- **R Integration**: Tests for R environment integration (mocked)

### 5. Web Interface Tests (`test_web_interface.py`)
- **Streamlit Components**: Tests for all Streamlit UI components
- **Data Display**: Tests for data visualization and metrics
- **Model Loading**: Tests for model loading and display
- **User Interaction**: Tests for user input handling
- **Error Handling**: Tests for graceful error handling

## Running Tests

### Prerequisites
Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Run All Tests
```bash
# Using pytest directly
pytest tests/ -v

# Using the test runner script
python tests/test_runner.py

# With coverage reporting
pytest tests/ -v --cov=grids --cov-report=html
```

### Run Specific Test Files
```bash
# Run specific test file
python tests/test_runner.py test_data_processing.py

# Run specific test class
pytest tests/test_data_processing.py::TestProcessInput -v

# Run specific test method
pytest tests/test_data_processing.py::TestProcessInput::test_process_csv_input -v
```

### Run Tests with Coverage
```bash
# Generate coverage report
pytest tests/ -v --cov=grids --cov-report=term-missing --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

## Test Fixtures

The `conftest.py` file provides common test fixtures:

- `temp_dir`: Temporary directory for test files
- `sample_csv_data`: Sample CSV data for testing
- `sample_processed_data`: Sample processed data
- `mock_uploaded_file`: Mock uploaded file object
- `test_db_path`: Path to test database
- `sample_model_data`: Sample data for model testing
- `cache_dir`: Temporary cache directory
- `setup_test_environment`: Automatic test environment setup

## Mocking Strategy

The tests use extensive mocking to isolate components:

1. **R Environment**: All R interactions are mocked to avoid R dependency
2. **File System**: File operations are mocked or use temporary directories
3. **Streamlit**: All Streamlit components are mocked for headless testing
4. **Database**: SQLite operations use temporary databases
5. **External Dependencies**: External libraries are mocked where appropriate

## Test Categories

### Unit Tests
- Individual function and method testing
- Isolated component testing
- Mock-based dependency isolation

### Integration Tests
- Component interaction testing
- Data flow testing
- End-to-end workflow testing

### Error Handling Tests
- Exception scenario testing
- Edge case handling
- Graceful failure testing

## Coverage Goals

The test suite aims for:
- **Line Coverage**: >90%
- **Branch Coverage**: >85%
- **Function Coverage**: >95%

## Continuous Integration

Tests are designed to run in CI/CD environments:
- No external dependencies (R, database servers)
- Fast execution (<30 seconds for full suite)
- Deterministic results
- Clear error reporting

## Adding New Tests

When adding new functionality:

1. **Create corresponding test file** if it doesn't exist
2. **Add unit tests** for individual functions
3. **Add integration tests** for component interactions
4. **Update fixtures** in `conftest.py` if needed
5. **Ensure proper mocking** for external dependencies
6. **Maintain test isolation** - tests should not depend on each other

## Test Best Practices

1. **Descriptive Names**: Use clear, descriptive test names
2. **Arrange-Act-Assert**: Structure tests with clear sections
3. **Single Responsibility**: Each test should test one thing
4. **Proper Setup/Teardown**: Use fixtures for common setup
5. **Mock External Dependencies**: Avoid real external calls
6. **Test Edge Cases**: Include boundary and error conditions
7. **Maintain Test Data**: Keep test data realistic but minimal

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `grids` package is in Python path
2. **Mock Issues**: Check mock setup and patching
3. **Temporary Files**: Ensure proper cleanup in fixtures
4. **R Dependencies**: All R calls should be mocked

### Debugging Tests

```bash
# Run with detailed output
pytest tests/ -v -s

# Run with debugger
pytest tests/ --pdb

# Run specific failing test
pytest tests/test_file.py::test_method -v -s --pdb
```

## Performance Considerations

- Tests should run quickly (<30 seconds total)
- Use temporary directories for file operations
- Mock expensive operations (R computations, database queries)
- Avoid real network calls or external services
- Use appropriate test data sizes (small but representative) 