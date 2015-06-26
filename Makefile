#
#

CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
LOGISTIC := bin/r_logistic
LEAST_SQUARE := bin/r_least_square

SRCEXT := cc
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -O3 -fopenmp
LIB := -fopenmp
INC := -I include

all:	$(LOGISTIC) $(LEAST_SQUARE)

# build the logistic app
$(LOGISTIC): build/algebra.o build/logistic.o build/matrices.o build/test_logistic.o
	@echo " Linking..."
	@echo " $(CC) $^ -o $(LOGISTIC) $(LIB)"; $(CC) $^ -o $(LOGISTIC) $(LIB)

# build the logistic app
$(LEAST_SQUARE): build/algebra.o build/least_square.o build/matrices.o build/test_least_square.o
	@echo " Linking..."
	@echo " $(CC) $^ -o $(LEAST_SQUARE) $(LIB)"; $(CC) $^ -o $(LEAST_SQUARE) $(LIB)


$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(LOGISTIC)"; $(RM) -r $(BUILDDIR) $(LOGISTIC)
	@echo " $(RM) -r $(BUILDDIR) $(LEAST_SQUARE)"; $(RM) -r $(BUILDDIR) $(LEAST_SQUARE)

