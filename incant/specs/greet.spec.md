---
target: wat
---
# Greet user

## Scenario: basic greeting
Given the program starts
When the user enters "Ben"
Then output "hello Ben\n"

## Scenario: empty name
Given the program starts
When the user enters ""
Then output "hello \n"

## Scenario: longer name
Given the program starts
When the user enters "Alexander"
Then output "hello Alexander\n"
