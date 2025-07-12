# Test Suite Documentation

This directory contains comprehensive unit and integration tests for the Model Test Bench application, with a focus on the HuggingFace dataset download functionality.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest fixtures and configuration
├── test_corpus_huggingface.py  # Unit tests for HuggingFace functionality
├── test_corpus_api_integration.py  # Integration tests for API endpoints
└── README.md                   # This file
```

## Test Categories

### Unit Tests (`test_corpus_huggingface.py`)
- **Purpose**: Test individual functions and components in isolation
- **Focus**: HuggingFace dataset download logic, error handling, data processing
- **Coverage**: 
  - Dataset loading and validation
  - Text extraction and processing
  - File operations
  - Database operations
  - Error handling scenarios

### Integration Tests (`test_corpus_api_integration.py`)
- **Purpose**: Test the complete API endpoints and their interactions
- **Focus**: End-to-end functionality through FastAPI client
- **Coverage**:
  - API endpoint responses
  - Database integration
  - File system operations
  - Error responses

## Running Tests

### Prerequisites
1. Ensure you're in the project root directory
2. Activate the virtual environment: `.venv`
3. Install dependencies: `uv sync`

### Basic Test Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run only unit tests
uv run pytest tests/test_corpus_huggingface.py -v -m unit

# Run only integration tests
uv run pytest tests/test_corpus_api_integration.py -v -m integration

# Run tests with coverage
uv run pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test class
uv run pytest tests/test_corpus_huggingface.py::TestHuggingFaceCorpusCreation -v

# Run specific test method
uv run pytest tests/test_corpus_huggingface.py::TestHuggingFaceCorpusCreation::test_successful_corpus_creation_basic -v
```

### Using the Test Runner Script

```bash
# Run the comprehensive test suite
python run_tests.py
```

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

### Database Fixtures
- `temp_db`: Creates a temporary SQLite database for testing
- `client`: FastAPI test client with database dependency overridden

### Mock Fixtures
- `mock_datasets`: Mocks the HuggingFace datasets library
- `mock_aiofiles`: Mocks async file operations
- `mock_os_makedirs`: Mocks directory creation
- `mock_os_path_exists`: Mocks file existence checks

### Sample Data Fixtures
- `sample_dataset`: Sample HuggingFace Dataset for testing
- `sample_dataset_dict`: Sample DatasetDict with multiple splits
- `sample_iterable_dataset`: Sample IterableDataset for error testing
- `valid_corpus_data`: Valid corpus creation data

## Test Scenarios Covered

### Success Scenarios
1. **Basic corpus creation**: Standard HuggingFace dataset download
2. **Config-based creation**: Dataset with specific configuration
3. **DatasetDict handling**: Multi-split dataset resolution
4. **File operations**: Proper file creation and content writing
5. **Database integration**: Corpus record creation and retrieval

### Error Scenarios
1. **Iterable datasets**: Rejection of streaming datasets
2. **Invalid splits**: Non-existent split in DatasetDict
3. **Missing columns**: Text column not found in dataset
4. **Config errors**: Missing required configuration
5. **File system errors**: Directory/file creation failures
6. **Database errors**: Rollback on failure

### Edge Cases
1. **Empty/None texts**: Filtering of empty content
2. **None column names**: Handling datasets with no column info
3. **Large datasets**: Performance considerations
4. **Special characters**: Unicode and special character handling

## Test Data

Tests use synthetic data to avoid external dependencies:
- Sample datasets are created in-memory using the HuggingFace datasets library
- No actual HuggingFace Hub downloads occur during testing
- File operations are mocked to avoid disk I/O

## Coverage Goals

The test suite aims for:
- **Function coverage**: All code paths in the HuggingFace functionality
- **Branch coverage**: All conditional logic and error handling
- **Integration coverage**: Complete API endpoint testing
- **Error coverage**: All exception scenarios

## Best Practices

1. **Isolation**: Each test is independent and doesn't rely on others
2. **Mocking**: External dependencies are mocked to ensure test reliability
3. **Cleanup**: Temporary resources are properly cleaned up
4. **Descriptive names**: Test names clearly describe what they're testing
5. **Documentation**: Each test has a clear docstring explaining its purpose

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the project root
2. **Database errors**: Check that the temporary database is being created properly
3. **Mock issues**: Verify that mocks are properly configured in fixtures
4. **Async errors**: Ensure proper async/await usage in test methods

### Debug Mode

To run tests in debug mode with more verbose output:

```bash
uv run pytest tests/ -v -s --tb=long
```

### Test Development

When adding new tests:
1. Follow the existing naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Add proper docstrings and comments
4. Ensure tests are isolated and don't have side effects
5. Update this README if adding new test categories 