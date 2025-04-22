
all: 
	@$(MAKE) --no-print-directory -f make/mac.make

clean:
	@$(MAKE) --no-print-directory -f make/mac.make clean

.PHONY: all clean

# Handy debugging for make variables
print-%:
	@echo '$*=$($*)'