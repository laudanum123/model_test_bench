# Template Structure Documentation

## Overview

The Model Test Bench application has been refactored from a single large `index.html` file into a modular, component-based structure using Jinja2 templates. This approach provides better maintainability, reusability, and separation of concerns.

## File Structure

```
app/templates/
├── base.html              # Base template with common layout and styles
├── index.html             # Simple redirect to dashboard
├── dashboard.html         # Dashboard page
├── corpus.html           # Corpus management page
├── questions.html        # Questions viewing page
├── evaluation.html       # Evaluation list page
└── new_evaluation.html   # New evaluation creation page
```

## Template Architecture

### Base Template (`base.html`)
- **Purpose**: Contains the common HTML structure, CSS styles, and navigation
- **Features**:
  - Responsive sidebar navigation
  - Bootstrap 5 and Font Awesome integration
  - Global JavaScript utilities (`showAlert`, `getStatusColor`, etc.)
  - Template blocks for extensibility:
    - `{% block title %}` - Page title
    - `{% block content %}` - Main content area
    - `{% block modals %}` - Modal components
    - `{% block extra_css %}` - Additional CSS
    - `{% block extra_js %}` - Page-specific JavaScript

### Individual Page Templates

#### Dashboard (`dashboard.html`)
- **Purpose**: Overview page with statistics and recent evaluations
- **Features**:
  - Statistics cards (corpora, questions, evaluations, running)
  - Recent evaluations list
  - Extends base template

#### Corpus Management (`corpus.html`)
- **Purpose**: Manage corpora (upload files, load HuggingFace datasets)
- **Features**:
  - Corpus list with view/delete actions
  - Upload file modal
  - HuggingFace dataset integration
  - Extends base template with modal block

#### Questions (`questions.html`)
- **Purpose**: View questions for selected corpora
- **Features**:
  - Corpus selection dropdown
  - Questions display
  - Simple, focused interface

#### Evaluations (`evaluation.html`)
- **Purpose**: View and manage evaluations
- **Features**:
  - Evaluation list with status badges
  - Run/view actions
  - Performance metrics display

#### New Evaluation (`new_evaluation.html`)
- **Purpose**: Create new evaluations
- **Features**:
  - Comprehensive form for evaluation configuration
  - Model selection dropdowns
  - Corpus selection

## Benefits of This Structure

### 1. **Maintainability**
- Each page has its own template file
- Easier to locate and fix issues
- Clear separation of concerns

### 2. **Reusability**
- Common components in base template
- Consistent styling and navigation
- Shared JavaScript utilities

### 3. **Performance**
- Only load necessary JavaScript for each page
- Smaller individual file sizes
- Better caching potential

### 4. **Team Development**
- Multiple developers can work on different pages
- Reduced merge conflicts
- Clear ownership of components

### 5. **Testing**
- Easier to unit test individual components
- Isolated functionality
- Better error isolation

## Routing

The application uses FastAPI routing to serve different templates:

```python
@app.get("/")                    # Redirects to /dashboard
@app.get("/dashboard")           # Serves dashboard.html
@app.get("/corpus")              # Serves corpus.html
@app.get("/questions")           # Serves questions.html
@app.get("/evaluation")          # Serves evaluation.html
@app.get("/new-evaluation")      # Serves new_evaluation.html
```

## Navigation

Navigation is handled through standard HTML links rather than JavaScript-based section switching:

```html
<a class="nav-link" href="/dashboard">Dashboard</a>
<a class="nav-link" href="/corpus">Corpora</a>
<a class="nav-link" href="/questions">Questions</a>
<a class="nav-link" href="/evaluation">Evaluations</a>
<a class="nav-link" href="/new-evaluation">New Evaluation</a>
```

## JavaScript Organization

### Global Functions (in `base.html`)
- `showAlert(message, type)` - Display alerts
- `getStatusColor(status)` - Get Bootstrap color classes
- `loadAvailableModels()` - Load model configurations

### Page-Specific Functions
Each template contains only the JavaScript needed for that specific page, reducing the overall bundle size and improving performance.

## Migration from Single-Page Application

The original `index.html` was a single-page application (SPA) with JavaScript-based navigation. The new structure:

1. **Eliminates** the need for complex JavaScript navigation logic
2. **Improves** SEO with proper URLs
3. **Enables** browser back/forward functionality
4. **Reduces** initial page load time
5. **Simplifies** debugging and maintenance

## Future Enhancements

This modular structure enables future improvements:

1. **Component Libraries**: Create reusable template components
2. **Progressive Enhancement**: Add JavaScript features incrementally
3. **Caching Strategies**: Implement template caching
4. **Internationalization**: Add multi-language support
5. **Accessibility**: Improve screen reader support

## Best Practices Followed

1. **DRY Principle**: Common code in base template
2. **Separation of Concerns**: HTML, CSS, and JS properly separated
3. **Progressive Enhancement**: Works without JavaScript
4. **Responsive Design**: Mobile-friendly layout
5. **Accessibility**: Semantic HTML and ARIA labels
6. **Performance**: Minimal JavaScript per page 