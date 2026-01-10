"""
Service for uploading motorcycle spare parts from Excel to Supabase with embeddings.
"""
import pandas as pd
import time
from typing import List, Dict, Optional
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client
from app.utils.logger import logger


class SparePartsUploader:
    """Service for uploading spare parts with embeddings to Supabase."""
    
    def __init__(self):
        """Initialize the uploader with required clients."""
        if voyage_client is None:
            raise ValueError("Voyage AI client not initialized. Please configure VOYAGE_API_KEY.")
        if supabase_vector_client is None:
            raise ValueError("Supabase client not initialized. Please configure SUPABASE_URL and SUPABASE_KEY.")
        
        self.voyage_client = voyage_client
        self.supabase_client = supabase_vector_client.client
    
    def read_excel_file(self, file_content: bytes) -> pd.DataFrame:
        """
        Read Excel file content and return as DataFrame.
        
        Args:
            file_content: Bytes content of the Excel file
            
        Returns:
            DataFrame containing the Excel data
        """
        from io import BytesIO
        
        logger.info("Reading Excel file")
        # Wrap bytes in BytesIO to avoid deprecation warning
        df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
        logger.info(f"Found {len(df)} rows with columns: {list(df.columns)}")
        return df
    
    def prepare_spare_part_data(self, row: pd.Series) -> Optional[Dict]:
        """
        Convert Excel row to spare part dictionary.
        Adjust column names based on your actual Excel structure.
        
        Args:
            row: Pandas Series representing a row from the Excel file
            
        Returns:
            Dictionary with spare part data, or None if invalid
        """
        part_data = {}
        
        # Helper function to get column value with flexible matching
        def get_column_value(possible_names: List[str], default=None):
            for name in possible_names:
                # Try exact match first
                if name in row.index:
                    val = row[name]
                    return None if pd.isna(val) else val
                # Try case-insensitive match
                for col in row.index:
                    if col.lower() == name.lower():
                        val = row[col]
                        return None if pd.isna(val) else val
            return default
        
        # Extract all fields matching the motorcycle_spareparts table structure
        part_data['partnumber'] = str(get_column_value(['partnumber', 'Part Number', 'PartNumber', 'part_number', 'PARTNUMBER'], '') or '')
        part_data['oemnumber'] = str(get_column_value(['oemnumber', 'OEM Number', 'OEMNumber', 'oem_number', 'OEMNUMBER'], '') or '')
        
        part_name = get_column_value(
            ['partname', 'Part Name', 'PartName', 'part_name', 'Name', 'name', 'Part', 'PART NAME'],
            'Unknown Part'
        )
        
        if not part_name or part_name == 'Unknown Part':
            logger.warning(f"Row {row.name} missing part name, skipping")
            return None
        
        part_data['partname'] = str(part_name)
        part_data['part_name'] = str(part_name)  # Keep for embedding text generation
        
        part_data['brand'] = str(get_column_value(['brand', 'Brand', 'Manufacturer', 'manufacturer', 'Make', 'MAKE', 'BRAND'], '') or '')
        part_data['category'] = str(get_column_value(['category', 'Category', 'CATEGORY', 'cat'], '') or '')
        part_data['make'] = str(get_column_value(['make', 'Make', 'MAKE', 'vehicle_make'], '') or '')
        part_data['model'] = str(get_column_value(['model', 'Model', 'Vehicle Model', 'vehicle_model', 'MODEL'], '') or '')
        
        # Engine capacity
        engine_cap = get_column_value(['enginecapacitycc', 'Engine Capacity CC', 'engine_capacity_cc', 'EngineCapacityCC'], 0)
        try:
            part_data['enginecapacitycc'] = int(float(engine_cap)) if engine_cap else 0
        except (ValueError, TypeError):
            part_data['enginecapacitycc'] = 0
        
        # Region
        part_data['region'] = str(get_column_value(['region', 'Region', 'REGION'], '') or '')
        
        # Retail price
        retail_price = get_column_value(['retailpricevnd', 'Retail Price VND', 'retail_price_vnd', 'RetailPriceVND'], 0)
        try:
            if isinstance(retail_price, str):
                retail_price = retail_price.replace(',', '').replace('₫', '').replace('VND', '').replace('vnd', '').strip()
            part_data['retailpricevnd'] = int(float(retail_price)) if retail_price else 0
        except (ValueError, TypeError):
            part_data['retailpricevnd'] = 0
        
        # Cost price
        cost_price = get_column_value(['costpricevnd', 'Cost Price VND', 'cost_price_vnd', 'CostPriceVND'], 0)
        try:
            if isinstance(cost_price, str):
                cost_price = cost_price.replace(',', '').replace('₫', '').replace('VND', '').replace('vnd', '').strip()
            part_data['costpricevnd'] = int(float(cost_price)) if cost_price else 0
        except (ValueError, TypeError):
            part_data['costpricevnd'] = 0
        
        # Repair code and name
        part_data['repaircode'] = str(get_column_value(['repaircode', 'Repair Code', 'repair_code', 'RepairCode'], '') or '')
        part_data['repairname'] = str(get_column_value(['repairname', 'Repair Name', 'repair_name', 'RepairName'], '') or '')
        
        # Total repair price
        total_repair_price = get_column_value(['totalrepairpricevnd', 'Total Repair Price VND', 'total_repair_price_vnd', 'TotalRepairPriceVND'], 0)
        try:
            if isinstance(total_repair_price, str):
                total_repair_price = total_repair_price.replace(',', '').replace('₫', '').replace('VND', '').replace('vnd', '').strip()
            part_data['totalrepairpricevnd'] = int(float(total_repair_price)) if total_repair_price else 0
        except (ValueError, TypeError):
            part_data['totalrepairpricevnd'] = 0
        
        # Legacy fields for embedding text generation (backward compatibility)
        part_data['part_description'] = ''
        part_data['vehicle_type'] = 'motorcycle'
        
        return part_data
    
    def create_embedding_text(self, part: Dict) -> str:
        """
        Create rich text representation for embedding generation.
        
        Args:
            part: Dictionary containing spare part data
            
        Returns:
            Formatted text string for embedding
        """
        # Use partname or part_name (backward compatibility)
        part_name = part.get('partname') or part.get('part_name', '')
        text_parts = [f"Part: {part_name}"]
        
        text_parts.append(f"Vehicle: {part.get('vehicle_type', 'motorcycle')}")
        
        if part.get('brand'):
            text_parts.append(f"Brand: {part['brand']}")
        if part.get('category'):
            text_parts.append(f"Category: {part['category']}")
        if part.get('make'):
            text_parts.append(f"Make: {part['make']}")
        if part.get('model'):
            text_parts.append(f"Model: {part['model']}")
        if part.get('repairname'):
            text_parts.append(f"Repair: {part['repairname']}")
        
        return " | ".join(text_parts)
    
    def verify_table_structure(self, table_name: str = "motorcycle_spareparts") -> bool:
        """
        Verify that the table exists and has the correct structure.
        Checks for motorcycle_spareparts table structure (partname, brand, etc.) or old structure (content).
        
        Args:
            table_name: Name of the table to verify
            
        Returns:
            True if table structure is correct, False otherwise
        """
        try:
            # Try to query for the new table structure (motorcycle_spareparts with partname column)
            result = self.supabase_client.table(table_name).select('id,partname').limit(1).execute()
            logger.info(f"Table {table_name} exists and has correct structure (motorcycle_spareparts schema)")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            error_dict = str(e) if isinstance(e, dict) else ''
            # Check for schema/column errors
            if any(keyword in error_msg for keyword in ['could not find', 'does not exist', 'pgrst204', 'partname', 'content']):
                logger.error(f"Table {table_name} structure issue detected: {e}")
                logger.error("=" * 80)
                logger.error("ACTION REQUIRED:")
                logger.error("1. Run the SQL setup script: supabase_spareparts_setup.sql")
                logger.error("2. Refresh PostgREST schema cache:")
                logger.error("   - Go to Supabase Dashboard -> Settings -> API")
                logger.error("   - Click 'Reload Schema' or 'Refresh Schema Cache'")
                logger.error("   - Wait a few seconds for the cache to refresh")
                logger.error("=" * 80)
                return False
            else:
                # Table might not exist at all
                logger.warning(f"Could not verify table structure: {e}")
                logger.warning("The table might not exist. Please run supabase_spareparts_setup.sql")
                return False
    
    def upload_batch(
        self,
        parts: List[Dict],
        batch_size: int = 50,
        table_name: str = "spareparts_embeddings"
    ) -> Dict:
        """
        Upload spare parts with embeddings in batches.
        
        Args:
            parts: List of spare part dictionaries
            batch_size: Number of items to process per batch (Voyage AI allows up to 128)
            table_name: Name of the Supabase table to upload to
            
        Returns:
            Dictionary with upload statistics
        """
        total_parts = len(parts)
        logger.info(f"Starting upload of {total_parts} parts in batches of {batch_size}")
        
        # Verify table structure first
        if not self.verify_table_structure(table_name):
            return {
                'total_parts': total_parts,
                'uploaded': 0,
                'failed': total_parts,
                'errors': [f"Table {table_name} structure verification failed. Please run supabase_spareparts_setup.sql and refresh schema cache."]
            }
        
        uploaded_count = 0
        failed_count = 0
        errors = []
        
        for i in range(0, total_parts, batch_size):
            batch = parts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_parts + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            try:
                # Generate embedding texts
                texts = [self.create_embedding_text(part) for part in batch]
                
                # Generate embeddings using Voyage AI with retry logic for rate limits
                logger.info(f"Generating embeddings for batch {batch_num}")
                max_retries = 3
                embeddings = None
                
                for retry in range(max_retries):
                    try:
                        embeddings = self.voyage_client.generate_embeddings_batch(
                            texts=texts,
                            input_type="document"
                        )
                        break  # Success, exit retry loop
                    except Exception as e:
                        error_msg = str(e).lower()
                        if 'rate limit' in error_msg or 'rpm' in error_msg or 'tpm' in error_msg or 'payment method' in error_msg:
                            if retry < max_retries - 1:
                                wait_time = (retry + 1) * 60  # Wait 60, 120, 180 seconds
                                logger.warning(f"Voyage AI rate limit hit. Waiting {wait_time} seconds before retry {retry + 1}/{max_retries}...")
                                logger.warning("Note: Add payment method at https://dashboard.voyageai.com/ to increase rate limits")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"Voyage AI rate limit exceeded after {max_retries} retries. Please add payment method or try again later.")
                                raise
                        else:
                            raise  # Re-raise if it's not a rate limit error
                
                if embeddings is None:
                    raise ValueError("Failed to generate embeddings after retries")
                
                if len(embeddings) != len(batch):
                    raise ValueError(f"Mismatch: {len(batch)} parts but {len(embeddings)} embeddings")
                
                # Prepare data for batch insertion - match motorcycle_spareparts table structure
                insert_data = []
                for part, embedding in zip(batch, embeddings):
                    # Prepare the data structure matching motorcycle_spareparts table columns
                    insert_row = {
                        'partnumber': part.get('partnumber', ''),
                        'oemnumber': part.get('oemnumber', ''),
                        'partname': part.get('partname', part.get('part_name', '')),
                        'brand': part.get('brand', ''),
                        'category': part.get('category', ''),
                        'make': part.get('make', ''),
                        'model': part.get('model', ''),
                        'enginecapacitycc': part.get('enginecapacitycc', 0),
                        'region': part.get('region', ''),
                        'retailpricevnd': part.get('retailpricevnd', 0),
                        'costpricevnd': part.get('costpricevnd', 0),
                        'repaircode': part.get('repaircode', ''),
                        'repairname': part.get('repairname', ''),
                        'totalrepairpricevnd': part.get('totalrepairpricevnd', 0),
                        'embedding': embedding  # 1024-dimensional vector
                    }
                    insert_data.append(insert_row)
                
                # Try batch insert first (faster)
                batch_inserted = 0
                try:
                    logger.info(f"Attempting batch insert for batch {batch_num}")
                    result = self.supabase_client.table(table_name).insert(insert_data).execute()
                    
                    if result.data:
                        batch_inserted = len(result.data)
                        logger.info(f"Batch insert successful: {batch_inserted} items")
                    else:
                        raise ValueError("Batch insert returned no data")
                        
                except Exception as e:
                    error_msg = str(e) if isinstance(e, str) else str(e).lower()
                    error_dict = str(e) if isinstance(e, dict) else ''
                    # Check for schema/column errors more comprehensively
                    # Check for schema/column errors
                    is_schema_error = (
                        any(keyword in error_msg for keyword in ['could not find', 'does not exist', 'pgrst204', 'column']) or
                        (isinstance(e, dict) and (
                            'PGRST204' in str(e.get('code', '')) or
                            any(keyword in str(e.get('message', '')).lower() for keyword in ['column', 'does not exist'])
                        )) or
                        (hasattr(e, 'code') and 'PGRST204' in str(e.code))
                    )
                    
                    if is_schema_error:
                        logger.error(f"Table structure error detected: {e}")
                        logger.error("=" * 80)
                        logger.error("SCHEMA ERROR - Column mismatch detected")
                        logger.error("SOLUTION:")
                        logger.error("1. Ensure you've run: supabase_spareparts_setup.sql")
                        logger.error("2. Refresh PostgREST schema cache:")
                        logger.error("   - Supabase Dashboard -> Settings -> API -> Reload Schema")
                        logger.error("3. Wait 30 seconds after refreshing, then retry")
                        logger.error("=" * 80)
                        # Re-raise to fail the batch
                        raise
                    else:
                        # Re-raise if it's a different error
                        raise
                
                if result.data and len(result.data) > 0:
                    batch_uploaded = len(result.data)
                    uploaded_count += batch_uploaded
                    logger.info(f"Batch {batch_num} uploaded successfully: {batch_uploaded} items")
                else:
                    logger.warning(f"Batch {batch_num} returned no data or all items failed")
                    failed_count += len(batch) - batch_inserted
                    if batch_inserted == 0:
                        errors.append(f"Batch {batch_num}: All items failed to upload. Check table structure and Supabase schema.")
                
                # Rate limiting: small delay between batches
                if i + batch_size < total_parts:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {str(e)}")
                failed_count += len(batch)
                errors.append(f"Batch {batch_num}: {str(e)}")
                continue
        
        return {
            'total_parts': total_parts,
            'uploaded': uploaded_count,
            'failed': failed_count,
            'errors': errors if errors else None
        }
    
    def process_excel_and_upload(
        self,
        file_content: bytes,
        batch_size: int = 50,
        table_name: str = "spareparts_embeddings"
    ) -> Dict:
        """
        Main function to process Excel and upload with embeddings.
        
        Args:
            file_content: Bytes content of the Excel file
            batch_size: Number of items to process per batch
            table_name: Name of the Supabase table to upload to
            
        Returns:
            Dictionary with processing and upload results
        """
        try:
            # Read Excel
            df = self.read_excel_file(file_content)
            
            if df.empty:
                return {
                    'success': False,
                    'error': 'Excel file is empty',
                    'stats': {}
                }
            
            # Prepare all parts
            logger.info("Preparing data from Excel")
            parts = []
            skipped_rows = []
            
            for idx, row in df.iterrows():
                try:
                    part = self.prepare_spare_part_data(row)
                    if part:
                        parts.append(part)
                    else:
                        skipped_rows.append(idx)
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {str(e)}")
                    skipped_rows.append(idx)
                    continue
            
            if not parts:
                return {
                    'success': False,
                    'error': 'No valid parts found in Excel file',
                    'stats': {
                        'total_rows': len(df),
                        'skipped_rows': skipped_rows
                    }
                }
            
            logger.info(f"Prepared {len(parts)} parts for upload (skipped {len(skipped_rows)} rows)")
            
            # Upload with embeddings
            upload_stats = self.upload_batch(parts, batch_size=batch_size, table_name=table_name)
            
            # Get total count from database
            try:
                result = self.supabase_client.table(table_name).select('id', count='exact').limit(1).execute()
                total_in_db = result.count if hasattr(result, 'count') else None
            except Exception as e:
                logger.warning(f"Could not get total count from database: {e}")
                total_in_db = None
            
            return {
                'success': True,
                'stats': {
                    **upload_stats,
                    'total_rows_in_excel': len(df),
                    'skipped_rows': skipped_rows,
                    'total_in_database': total_in_db
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel and uploading: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'stats': {}
            }