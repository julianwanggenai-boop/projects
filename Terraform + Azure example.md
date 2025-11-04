We’ll use **Terraform + Azure**, but the same principles apply to AWS, GCP, etc.

---

# 🧠 1. What Terraform Does

Terraform is an **Infrastructure as Code (IaC)** tool that lets you:

* **Define** your infrastructure in `.tf` files (like Azure VMs, Storage, AKS, etc.)
* **Plan** changes before applying them.
* **Apply** those changes automatically and reproducibly.
* **Track** real-world resources in its **state file**.

It’s **declarative** — you say *what* you want, not *how* to build it.

---

# 🚀 2. “Hello World” Example — Creating a Resource Group on Azure

### Project Structure

```
terraform-hello-world/
├── main.tf
├── variables.tf
└── outputs.tf
```

---

## 🗂️ `main.tf`

This is your main Terraform configuration file.

```hcl
# main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Create a simple Azure Resource Group (Hello World!)
resource "azurerm_resource_group" "hello_rg" {
  name     = "hello-world-rg"
  location = "eastus"
}
```

---

## ⚙️ `variables.tf` (optional for this example)

You can define variables here to make configurations reusable.

```hcl
# variables.tf
variable "location" {
  description = "Azure region to deploy to"
  type        = string
  default     = "eastus"
}
```

Then reference it in `main.tf`:

```hcl
location = var.location
```

---

## 📤 `outputs.tf`

Outputs show you useful info after deployment (like IPs, URLs, IDs).

```hcl
# outputs.tf
output "resource_group_name" {
  value = azurerm_resource_group.hello_rg.name
}
```

---

# 🧩 3. Initialize and Apply the Project

### Step 1: Install Terraform

👉 Download from [terraform.io/downloads](https://developer.hashicorp.com/terraform/downloads)

Then check version:

```bash
terraform -v
```

### Step 2: Log in to Azure

You’ll need to authenticate before Terraform can deploy.

```bash
az login
az account show
```

### Step 3: Initialize Terraform

This sets up providers and prepares the working directory.

```bash
terraform init
```

Output:

```
Initializing the backend...
Initializing provider plugins...
Terraform has been successfully initialized!
```

---

### Step 4: Preview the Plan

Terraform shows what will change (nothing is deployed yet):

```bash
terraform plan
```

Output:

```
Terraform will perform the following actions:
  + create azurerm_resource_group.hello_rg
```

---

### Step 5: Apply the Plan

Actually deploy your infrastructure to Azure:

```bash
terraform apply
```

Confirm with `yes` or use `-auto-approve`.

Output:

```
azurerm_resource_group.hello_rg: Creating...
azurerm_resource_group.hello_rg: Creation complete

Apply complete! Resources: 1 added, 0 changed, 0 destroyed.

Outputs:
resource_group_name = "hello-world-rg"
```

🎉 **Congratulations!**
You just created your first Azure resource — using code!

---

# 🔍 4. Verify in Azure

Check the new Resource Group:

```bash
az group list --query "[?name=='hello-world-rg']"
```

Or open the **Azure Portal → Resource Groups** — you’ll see it listed.

---

# 🧹 5. Destroy the Infrastructure

When you’re done, clean up all resources created by Terraform:

```bash
terraform destroy -auto-approve
```

Output:

```
azurerm_resource_group.hello_rg: Destroying...
Destroy complete! Resources: 1 destroyed.
```

Terraform updates its **state file** to keep everything in sync.

---

# ⚙️ 6. How Terraform Works Internally

Here’s what happens under the hood:

| Step | Action              | Description                                          |
| ---- | ------------------- | ---------------------------------------------------- |
| 1    | `terraform init`    | Installs provider plugins (e.g., AzureRM)            |
| 2    | `terraform plan`    | Compares your `.tf` code vs current Azure state      |
| 3    | `terraform apply`   | Creates/updates/deletes resources to match your code |
| 4    | `terraform state`   | Tracks what Terraform manages in a `.tfstate` file   |
| 5    | `terraform destroy` | Deletes all resources defined in your `.tf` files    |

Terraform’s **state file** acts like its “memory,” letting it know what exists and what should change.

---

# 💡 Bonus: Hello World on AWS or Local Machine

You can even use Terraform for **local provisioning** (not just cloud).

Example: create a local file called “hello.txt”.

```hcl
# localfile.tf
terraform {
  required_providers {
    local = {
      source = "hashicorp/local"
    }
  }
}

resource "local_file" "hello" {
  content  = "Hello, Terraform!"
  filename = "${path.module}/hello.txt"
}
```

Run:

```bash
terraform init
terraform apply -auto-approve
cat hello.txt
```

Output:

```
Hello, Terraform!
```

🎉 You just created a local file using Terraform!

---

# 🏁 Summary

✅ **Terraform Workflow**

1. Write `.tf` code (desired state)
2. `terraform init`
3. `terraform plan`
4. `terraform apply`
5. `terraform destroy` (optional cleanup)

✅ **Concepts**

* Declarative infrastructure
* State tracking
* Idempotent (safe to re-run)
* Provider-based (Azure, AWS, GCP, etc.)
* Modular & reusable

---
